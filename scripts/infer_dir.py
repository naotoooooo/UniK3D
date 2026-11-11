import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from unik3d.models import UniK3D
# resolved by name via eval(), as in infer.py
from unik3d.utils.camera import (MEI, OPENCV, Fisheye624, Pinhole, Spherical)
from unik3d.utils.visualization import colorize, save_file_ply

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def save(
    rgb,
    outputs,
    name,
    base_path,
    save_map=False,
    save_pointcloud=False,
    vmin=None,
    vmax=None,
    vmax_percentile=-1.0,
):
    # as infer.save, without the rays map
    os.makedirs(base_path, exist_ok=True)

    if save_map:
        depth = outputs["distance"].cpu().numpy()
        # depth = outputs["depth"].cpu().numpy()
        if vmax_percentile > 0.0:
            vmax = percentile_vmax(depth, vmax_percentile)
        # fixed vmin/vmax keeps colors comparable across images
        Image.fromarray(colorize(depth.squeeze(), vmin=vmin, vmax=vmax)).save(
            os.path.join(base_path, f"{name}_depth.jpg"), quality=95
        )

    if save_pointcloud:
        points = outputs["points"]
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))


def percentile_vmax(depth, p):
    # far end of the color range at the p-th percentile of the valid depths
    # (colorize's invalid < 1e-4 left out); anything farther saturates to
    # the darkest color instead of washing out the rest of the image
    valid = depth[depth >= 0.0001]
    return float(np.percentile(valid, p)) if valid.size else None


def collect_images(input_dir):
    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(EXTENSIONS)
    )


def elapsed(since):
    # GPU work is async: sync before reading the clock
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - since


def load_rgb(image_path):
    return torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1)


def network_input_shape(model, rgb_torch, camera):
    # run infer once, recording the (H, W) the network actually sees after
    # infer's pad + resize, which is what resolution_level controls
    shapes = []
    encode_decode = model.encode_decode

    def recording(inputs, *args, **kwargs):
        shapes.append(tuple(inputs["image"].shape[-2:]))
        return encode_decode(inputs, *args, **kwargs)

    model.encode_decode = recording
    try:
        model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)
    finally:
        del model.encode_decode
    return shapes[0]


def summarize(times):
    ms = np.asarray(times) * 1000.0
    return {
        "mean_ms": float(ms.mean()),
        "std_ms": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
        "median_ms": float(np.median(ms)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "fps": float(1000.0 / ms.mean()),
    }


def format_stats(s):
    return (
        f"mean {s['mean_ms']:.2f} ± {s['std_ms']:.2f} ms, "
        f"median {s['median_ms']:.2f} ms, "
        f"min {s['min_ms']:.2f} / max {s['max_ms']:.2f} ms, "
        f"{s['fps']:.2f} FPS"
    )


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description="Inference script over a directory of images",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image directory."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        default="./configs/eval/vitl.json",
        help="Path to config file. Please check ./configs/eval.",
    )
    parser.add_argument(
        "--camera-path",
        type=str,
        default=None,
        help="Path to camera parameters json file, applied to every image. See assets/demo for a few examples. The file needs a 'name' field with the camera model from unik3d/utils/camera.py and a 'params' field with the camera parameters as in the corresponding class docstring.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save outputs as (colorized) jpg."
    )
    parser.add_argument(
        "--save-ply", action="store_true", help="Save pointcloud as ply."
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=9,
        help="Resolution level in [0,10). Higher values means it will resize to larger resolution which increases details but decreases speed. Lower values lead to opposite.",
        choices=list(range(10)),
    )
    parser.add_argument(
        "--interpolation-mode",
        type=str,
        default="bilinear",
        help="Output interpolation.",
        choices=["nearest", "nearest-exact", "bilinear"],
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Distance in meters mapped to the brightest color. Pass a negative value to normalize per image instead.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=30.0,
        help="Distance in meters mapped to the darkest color. Pass a negative value to normalize per image instead.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=60.0,
        help="Far end of the color range at this percentile of each image's depths, overriding --vmax; farther depths saturate to the darkest color. Pass a negative value to use --vmax instead.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Untimed inferences on the first image before timing, to exclude CUDA init and kernel selection.",
    )
    args = parser.parse_args()

    # negative bounds fall back to infer.py's per-image min/max
    vmin = args.vmin if args.vmin >= 0.0 else None
    vmax = args.vmax if args.vmax >= 0.0 else None

    image_paths = collect_images(args.input)
    if not len(image_paths):
        raise FileNotFoundError(f"No image found in {args.input}")

    print("Torch version:", torch.__version__)
    version = args.config_file.split("/")[-1].split(".")[0]
    name = f"unik3d-{version}"

    start = time.perf_counter()
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    time_load = elapsed(start)
    print(f"Model load: {time_load:.2f}s")

    # same camera for every image, so build it once
    camera = None
    if args.camera_path is not None:
        with open(args.camera_path, "r") as f:
            camera_dict = json.load(f)

        params = torch.tensor(camera_dict["params"])
        camera_name = camera_dict["name"]
        assert camera_name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(camera_name)(params=params)

    # model.infer crops/resizes the camera in place, so every call gets a fresh copy
    fresh_camera = lambda: camera.clone() if camera is not None else None

    rgb_first = load_rgb(image_paths[0])
    infer_shape = network_input_shape(model, rgb_first, fresh_camera())
    for _ in range(args.warmup):
        model.infer(rgb=rgb_first, camera=fresh_camera(), normalize=True, rays=None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_infer, times_save, input_shapes = [], [], set()
    for i, image_path in enumerate(image_paths):
        rgb_torch = load_rgb(image_path)
        input_shapes.add(tuple(rgb_torch.shape[-2:]))

        # model.infer crops/resizes the camera in place, so hand it a fresh copy
        camera_image = camera.clone() if camera is not None else None

        start = time.perf_counter()
        outputs = model.infer(
            rgb=rgb_torch, camera=camera_image, normalize=True, rays=None
        )
        times_infer.append(elapsed(start))

        start = time.perf_counter()
        save(
            rgb_torch,
            outputs,
            name=image_path.split("/")[-1].split(".")[0],
            base_path=args.output,
            save_map=args.save,
            save_pointcloud=args.save_ply,
            vmin=vmin,
            vmax=vmax,
            vmax_percentile=args.vmax_percentile,
        )
        times_save.append(elapsed(start))

        print(
            f"[{i + 1}/{len(image_paths)}] {image_path}: "
            f"infer {times_infer[-1] * 1000:.2f}ms, save {times_save[-1] * 1000:.2f}ms"
        )

    environment = {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        # UniK3D.infer runs under fp16 autocast whenever CUDA is available
        "precision": "fp16 autocast" if device.type == "cuda" else "fp32",
        "model": name,
        "resolution_level": args.resolution_level,
        "interpolation_mode": args.interpolation_mode,
        "input_hw": sorted(input_shapes),
        "network_input_hw": infer_shape,
        "batch_size": 1,
        "num_images": len(image_paths),
        "warmup": args.warmup,
    }
    stats_infer, stats_save = summarize(times_infer), summarize(times_save)

    print("\nEnvironment:")
    for key, value in environment.items():
        print(f"  {key}: {value}")
    if len(input_shapes) > 1:
        print("  (inputs differ in size: network_input_hw is for the first image only)")
    print(
        "\nInfer (H2D + pre/post-process + network, excl. image decode): "
        f"{format_stats(stats_infer)}"
    )
    print(f"Save: {format_stats(stats_save)}")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "timing.json"), "w") as f:
        json.dump(
            {
                "environment": environment,
                "load_s": time_load,
                "infer": stats_infer,
                "save": stats_save,
                "per_image_ms": {
                    os.path.basename(p): {"infer": ti * 1000, "save": ts * 1000}
                    for p, ti, ts in zip(image_paths, times_infer, times_save)
                },
            },
            f,
            indent=2,
        )
