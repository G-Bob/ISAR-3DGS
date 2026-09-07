#!/usr/bin/env python3
"""Generate one GEO-equivalent encounter animation for every fixed-K subset.

The script is designed for the output of ``scripts/select_encounters.py``:

    selection_dir/subsets/k01.txt
    selection_dir/subsets/k02.txt
    ...
    selection_dir/selection_report.json
    selection_dir/fixed_k_selection.csv

Each K is loaded independently.  This is important because the optimized set
S*_K is not generally nested inside S*_{K+1}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


GEO_ALTITUDE_KM = 35_786.0


@dataclass
class Track:
    name: str
    frame_indices: np.ndarray
    positions_km: np.ndarray
    selected_local_indices: np.ndarray


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-12)


def _view_angle_path(encounter: Path) -> Path:
    for candidate in (encounter / "view_angle.npy", encounter / "Es" / "view_angle.npy"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No view_angle.npy found for {encounter}")


def _read_subsets(selection_dir: Path) -> dict[int, list[str]]:
    subsets: dict[int, list[str]] = {}
    for path in sorted((selection_dir / "subsets").glob("k*.txt")):
        match = re.fullmatch(r"k(\d+)", path.stem)
        if not match:
            continue
        count = int(match.group(1))
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) != count:
            raise ValueError(f"{path} contains {len(names)} names, expected K={count}")
        if len(set(names)) != count:
            raise ValueError(f"{path} contains duplicate encounter names")
        subsets[count] = names
    if not subsets:
        raise FileNotFoundError(f"No kXX.txt subset files found under {selection_dir / 'subsets'}")
    return subsets


def _read_selection_report(selection_dir: Path) -> dict:
    path = selection_dir / "selection_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_metrics(selection_dir: Path) -> dict[int, dict[str, str]]:
    path = selection_dir / "fixed_k_selection.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {int(row["encounter_count"]): row for row in csv.DictReader(stream)}


def _equivalent_range_profile(count: int, cpa_km: float, max_range_km: float) -> np.ndarray:
    """Straight-line symmetric range history, exactly bounded by max_range_km."""
    along_limit = math.sqrt(max_range_km**2 - cpa_km**2)
    along = np.linspace(-along_limit, along_limit, count)
    return np.sqrt(cpa_km**2 + along**2)


def _resample_rows(values: np.ndarray, output_count: int) -> np.ndarray:
    if len(values) == output_count:
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, output_count)
    columns = [np.interp(target, source, values[:, column]) for column in range(values.shape[1])]
    return np.column_stack(columns)


def _build_track(
    data_root: Path,
    encounter_name: str,
    frame_start: int,
    frames_per_encounter: int,
    animation_frames: int,
    cpa_km: float,
    max_range_km: float,
    selected_frames: dict[str, list[int]],
) -> Track:
    encounter = data_root / encounter_name
    geometry = np.asarray(np.load(_view_angle_path(encounter)), dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] < 6:
        raise ValueError(f"Expected [N,6+] view_angle.npy for {encounter}")
    stop = min(len(geometry), frame_start + frames_per_encounter)
    frame_indices = np.arange(frame_start, stop, dtype=int)
    if len(frame_indices) < 2:
        raise ValueError(f"{encounter_name} has fewer than two usable frames")

    kr = _unit(geometry[frame_indices, :3])
    kr_animation = _unit(_resample_rows(kr, animation_frames))
    ranges = _equivalent_range_profile(animation_frames, cpa_km, max_range_km)
    positions = kr_animation * ranges[:, None]

    chosen = np.asarray(selected_frames.get(encounter_name, []), dtype=int)
    chosen = chosen[(chosen >= frame_indices[0]) & (chosen <= frame_indices[-1])]
    if len(chosen):
        fractional = (chosen - frame_indices[0]) / max(frame_indices[-1] - frame_indices[0], 1)
        chosen_local = np.unique(np.rint(fractional * (animation_frames - 1)).astype(int))
    else:
        chosen_local = np.zeros(0, dtype=int)
    return Track(encounter_name, frame_indices, positions, chosen_local)


def _cube_faces(size: float) -> list[list[np.ndarray]]:
    half = size / 2.0
    vertices = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    return [[vertices[index] for index in face] for face in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))]


def _draw_fixed_os(axis, scale: float) -> None:
    axis.add_collection3d(
        Poly3DCollection(_cube_faces(0.13 * scale), facecolor="#f59e0b", edgecolor="#fde68a", alpha=0.95)
    )
    for sign in (-1, 1):
        panel = [[
            np.array([-0.13 * scale, sign * 0.10 * scale, 0.0]),
            np.array([ 0.13 * scale, sign * 0.10 * scale, 0.0]),
            np.array([ 0.13 * scale, sign * 0.48 * scale, 0.0]),
            np.array([-0.13 * scale, sign * 0.48 * scale, 0.0]),
        ]]
        axis.add_collection3d(
            Poly3DCollection(panel, facecolor="#3b82f6", edgecolor="#93c5fd", alpha=0.9)
        )
    length = 0.16 * scale
    axis.quiver(0, 0, 0, length, 0, 0, color="#ef4444", arrow_length_ratio=0.09, linewidth=1.2)
    axis.quiver(0, 0, 0, 0, length, 0, color="#22c55e", arrow_length_ratio=0.09, linewidth=1.2)
    axis.quiver(0, 0, 0, 0, 0, length, color="#60a5fa", arrow_length_ratio=0.09, linewidth=1.2)


def _short_name(name: str, ordinal: int) -> str:
    index_match = re.search(r"idx(\d+)", name)
    if index_match:
        return f"E{ordinal}: idx{int(index_match.group(1)):04d}"
    return f"E{ordinal}: {name[-18:]}"


def _configure_axis(axis, limit: float) -> None:
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-0.72 * limit, 0.72 * limit)
    axis.set_box_aspect((2.0, 2.0, 1.45))
    axis.set_xlabel("Target-centred X (km)", color="#dbeafe")
    axis.set_ylabel("Target-centred Y (km)", color="#dbeafe")
    axis.set_zlabel("Target-centred Z (km)", color="#dbeafe")
    axis.tick_params(colors="#9fb3c8")
    for item in (axis.xaxis, axis.yaxis, axis.zaxis):
        item.pane.set_facecolor((0.03, 0.08, 0.14, 0.75))
        item.pane.set_edgecolor((0.28, 0.40, 0.55, 0.45))
        item._axinfo["grid"]["color"] = (0.35, 0.48, 0.62, 0.22)


def _animate_k(
    count: int,
    tracks: list[Track],
    metric: dict[str, str] | None,
    output: Path,
    args: argparse.Namespace,
) -> dict:
    figure = plt.figure(figsize=(11.2, 7.4), dpi=args.dpi)
    figure.patch.set_facecolor("#07111f")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#07111f")
    _configure_axis(axis, args.max_range_km * 1.06)
    axis.view_init(elev=args.elevation_deg, azim=args.azimuth_deg)
    _draw_fixed_os(axis, args.max_range_km)
    axis.text(0, 0, 0.13 * args.max_range_km, "OS fixed pose\nGEO altitude 35,786 km", color="#fde68a", ha="center")

    colors = plt.colormaps["tab20"](np.linspace(0.0, 1.0, max(len(tracks), 2)))
    moving_points = []
    trails = []
    labels = []
    for ordinal, (track, color) in enumerate(zip(tracks, colors), start=1):
        positions = track.positions_km
        label = _short_name(track.name, ordinal)
        labels.append({"label": label, "encounter": track.name})
        axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, linewidth=1.4, alpha=0.45, label=label)
        if args.show_selected_frames and len(track.selected_local_indices):
            chosen = positions[track.selected_local_indices]
            axis.scatter(chosen[:, 0], chosen[:, 1], chosen[:, 2], color=color, s=9, alpha=0.7, marker="|")
        point, = axis.plot([], [], [], marker="o", markersize=6.5, color=color, linestyle="None")
        trail, = axis.plot([], [], [], color=color, linewidth=2.8, alpha=0.95)
        moving_points.append(point)
        trails.append(trail)

    metric_text = ""
    if metric:
        metric_text = (
            f"direction union = {float(metric['direction_union_pct']):.2f}%   "
            f"normalized λmin = {float(metric['normalized_minimum_eigenvalue']):.4f}"
        )
    axis.set_title(
        f"Independent fixed-K encounter set: K={count}\n{metric_text}",
        color="#f8fafc",
        pad=16,
    )
    columns = 1 if count <= 5 else 2
    axis.legend(loc="lower left", frameon=False, labelcolor="#dbeafe", fontsize=8, ncol=columns)
    status = axis.text2D(0.02, 0.965, "", transform=axis.transAxes, color="#f8fafc", va="top", family="monospace")

    def update(animation_index: int):
        begin = max(0, animation_index - args.trail_frames)
        for track, point, trail in zip(tracks, moving_points, trails):
            position = track.positions_km[animation_index]
            point.set_data_3d([position[0]], [position[1]], [position[2]])
            segment = track.positions_km[begin : animation_index + 1]
            trail.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
        progress = animation_index / max(args.animation_frames - 1, 1)
        equivalent_range = math.sqrt(
            args.cpa_km**2
            + ((2.0 * progress - 1.0) * math.sqrt(args.max_range_km**2 - args.cpa_km**2)) ** 2
        )
        status.set_text(
            f"K = {count}\n"
            f"encounters = {len(tracks)}\n"
            f"progress = {100.0 * progress:5.1f}%\n"
            f"equivalent R = {equivalent_range:6.2f} km\n"
            f"reference ω = {args.omega_deg_s:g} deg/s"
        )
        return [*moving_points, *trails, status]

    animation = FuncAnimation(
        figure,
        update,
        frames=args.animation_frames,
        interval=1000.0 / args.fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        animation.save(output, writer=PillowWriter(fps=args.fps))
    else:
        animation.save(output, writer=FFMpegWriter(fps=args.fps, bitrate=args.bitrate_kbps))
    plt.close(figure)
    return {
        "encounter_count": count,
        "output": str(output),
        "encounters": labels,
        "metric": metric,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--selection_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--counts", nargs="+", type=int, help="Default: every available kXX.txt")
    parser.add_argument("--frames_per_encounter", type=int, default=120)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--animation_frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--trail_frames", type=int, default=28)
    parser.add_argument("--cpa_km", type=float, default=24.0)
    parser.add_argument("--max_range_km", type=float, default=100.0)
    parser.add_argument("--omega_deg_s", type=float, default=3.6)
    parser.add_argument("--show_selected_frames", action="store_true", default=True)
    parser.add_argument("--hide_selected_frames", action="store_false", dest="show_selected_frames")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--bitrate_kbps", type=int, default=2400)
    parser.add_argument("--elevation_deg", type=float, default=24.0)
    parser.add_argument("--azimuth_deg", type=float, default=-58.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.selection_dir = args.selection_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not (0.0 < args.cpa_km < args.max_range_km):
        raise ValueError("Require 0 < cpa_km < max_range_km")
    if args.animation_frames < 2:
        raise ValueError("animation_frames must be at least 2")

    subsets = _read_subsets(args.selection_dir)
    counts = sorted(set(args.counts or subsets))
    missing = [count for count in counts if count not in subsets]
    if missing:
        raise ValueError(f"No independently optimized subset files for K={missing}")
    report = _read_selection_report(args.selection_dir)
    selected_frames = report.get("selected_training_frames", {})
    metrics = _read_metrics(args.selection_dir)

    results = []
    for count in counts:
        tracks = [
            _build_track(
                args.data_root,
                name,
                args.frame_start,
                args.frames_per_encounter,
                args.animation_frames,
                args.cpa_km,
                args.max_range_km,
                selected_frames,
            )
            for name in subsets[count]
        ]
        output = args.out_dir / f"k{count:02d}_geo_encounters.{args.format}"
        print(f"Rendering K={count}: {', '.join(subsets[count])}", flush=True)
        results.append(_animate_k(count, tracks, metrics.get(count), output, args))

    manifest = {
        "method": "independent_fixed_k_geo_equivalent_animation",
        "subsets_are_nested": False,
        "data_root": str(args.data_root),
        "selection_dir": str(args.selection_dir),
        "cpa_km": args.cpa_km,
        "maximum_range_km": args.max_range_km,
        "reference_angular_speed_deg_s": args.omega_deg_s,
        "outputs": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "animation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(results)} animations under {args.out_dir}")


if __name__ == "__main__":
    main()
