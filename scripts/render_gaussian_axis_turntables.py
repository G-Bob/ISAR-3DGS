from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.array([1.35, -1.65, 1.05], dtype=np.float32)
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def render_frame(
    xyz: np.ndarray,
    support: np.ndarray,
    marker_radius: np.ndarray,
    axis: str,
    angle: float,
    extent: float,
    size: int,
) -> Image.Image:
    rot = rotation(axis, angle)
    points = xyz @ rot.T
    right, up, forward = camera_basis()
    u, v, depth = points @ right, points @ up, points @ forward
    margin = 74
    scale = (size - 2 * margin) / (2.0 * extent)
    px = size * 0.5 + u * scale
    py = size * 0.5 - v * scale
    colors = (matplotlib.colormaps["magma"](support)[:, :3] * 255).astype(np.uint8)
    image = Image.new("RGB", (size, size), (3, 3, 7))
    draw = ImageDraw.Draw(image, "RGBA")
    for idx in np.argsort(depth):
        if not (-10 <= px[idx] <= size + 10 and -10 <= py[idx] <= size + 10):
            continue
        radius = int(marker_radius[idx])
        color = tuple(int(x) for x in colors[idx]) + (210,)
        draw.ellipse(
            (px[idx] - radius, py[idx] - radius, px[idx] + radius, py[idx] + radius),
            fill=color,
        )

    origin = np.zeros(3, dtype=np.float32)
    axis_length = extent * 0.30
    endpoints = np.eye(3, dtype=np.float32) * axis_length
    projected = []
    for point in np.vstack((origin, endpoints)) @ rot.T:
        projected.append((size * 0.5 + point @ right * scale, size * 0.5 - point @ up * scale))
    labels = (("X", (255, 70, 70, 255)), ("Y", (80, 255, 110, 255)), ("Z", (70, 150, 255, 255)))
    for i, (label, color) in enumerate(labels, start=1):
        draw.line((projected[0], projected[i]), fill=color, width=4)
        draw.text((projected[i][0] + 5, projected[i][1] - 8), label, fill=color)
    degree = int(round(math.degrees(angle))) % 360
    draw.text((18, 16), f"K=10 Radar-GS | q >= 0.3 | rotation about {axis.upper()} axis | {degree:03d} deg", fill=(245, 245, 245, 255))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--q_threshold", type=float, default=0.3)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"]
    support = torch.sigmoid(state["support_logits"]).numpy()
    keep = support >= args.q_threshold
    xyz = state["means"].numpy()[keep]
    support = support[keep]
    scales = state["log_scales"].exp().numpy()[keep]
    center = np.median(xyz, axis=0)
    xyz = xyz - center
    extent = float(np.quantile(np.linalg.norm(xyz, axis=1), 0.995) * 1.10)
    marker_radius = np.clip(1.2 + 17.0 * np.sqrt(scales[:, :2].mean(axis=1)), 2.0, 5.0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for axis in ("x", "z"):
        frame_dir = out / f"rotate_{axis}_frames"
        frame_dir.mkdir(exist_ok=True)
        frames = []
        for index in range(args.frames):
            angle = 2.0 * math.pi * index / args.frames
            frame = render_frame(xyz, support, marker_radius, axis, angle, extent, args.size)
            frame.save(frame_dir / f"frame_{index:04d}.png")
            frames.append(frame)
        gif = out / f"k10_gaussians_rotate_{axis}.gif"
        frames[0].save(
            gif,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / args.fps),
            loop=0,
            optimize=False,
            disposal=2,
        )
        outputs[axis] = str(gif)
    manifest = {
        "checkpoint": args.checkpoint,
        "q_threshold": args.q_threshold,
        "retained_gaussians": int(len(xyz)),
        "center_m": center.tolist(),
        "fixed_extent_m": extent,
        "frames": args.frames,
        "fps": args.fps,
        "outputs": outputs,
    }
    (out / "animation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
