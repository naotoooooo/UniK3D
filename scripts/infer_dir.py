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
):
    # as infer.save, without the rays map
    os.makedirs(base_path, exist_ok=True)

    if save_map:
        depth = outputs["distance"].cpu().numpy()
        depth = outputs["depth"].cpu().numpy()
        # fixed vmin/vmax keeps colors comparable across images
        Image.fromarray(colorize(depth.squeeze(), vmin=vmin, vmax=vmax)).save(
            os.path.join(base_path, f"{name}_depth.png")
        )

    if save_pointcloud:
        points = outputs["points"]
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))


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
        "--save", action="store_true", help="Save outputs as (colorized) png."
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

    times_infer, times_save = [], []
    for i, image_path in enumerate(image_paths):
        rgb = np.array(Image.open(image_path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

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
        )
        times_save.append(elapsed(start))

        print(
            f"[{i + 1}/{len(image_paths)}] {image_path}: "
            f"infer {times_infer[-1]:.2f}s, save {times_save[-1]:.2f}s"
        )

    print(
        f"\nTotal for {len(image_paths)} images: "
        f"load {time_load:.2f}s, "
        f"infer {sum(times_infer):.2f}s (mean {np.mean(times_infer):.2f}s), "
        f"save {sum(times_save):.2f}s (mean {np.mean(times_save):.2f}s)"
    )
