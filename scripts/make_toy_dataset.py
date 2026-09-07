from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small GEIST-layout smoke-test dataset")
    parser.add_argument("--out", default="toy_data/starlink_toy_0001")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.out)
    image_dir = root / "isar_images_noaxis"
    image_dir.mkdir(parents=True, exist_ok=True)
    geometry = []
    yy, xx = np.mgrid[: args.size, : args.size]
    center = (args.size - 1) / 2.0
    scatterers = np.array(
        [[-0.55, -0.15, 1.0], [-0.2, 0.2, 0.7], [0.15, 0.0, 0.9], [0.55, 0.25, 0.65]]
    )
    for index in range(args.frames):
        angle = np.deg2rad(index * 3.0)
        kr = np.array([np.sin(angle), 0.0, np.cos(angle)])
        kd = np.array([np.cos(angle), 0.0, -np.sin(angle)])
        geometry.append(np.concatenate((kr, kd)))
        image = np.zeros((args.size, args.size), dtype=np.float32)
        for x_position, z_position, amplitude in scatterers:
            point = np.array([x_position, 0.0, z_position])
            row = center - np.dot(kr, point) * args.size / 3.0
            col = center + np.dot(kd, point) * args.size / 3.0
            image += amplitude * np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / 3.0)
        image /= image.max()
        Image.fromarray(np.uint8(image * 255), mode="L").save(image_dir / f"isar_{index:05d}.png")
    np.save(root / "view_angle.npy", np.asarray(geometry, dtype=np.float32))
    print(root)


if __name__ == "__main__":
    main()

