#!/usr/bin/env python3
"""Create a transformed copy of a reference GLB without modifying the source.

The default rotation converts the current GT/CAD coordinate convention into
the coordinate convention used by the original Radar-GS prediction PLY:

    x_pred = -x_gt
    y_pred =  z_gt
    z_pred =  y_gt

The matrix is its own inverse, but the direction is stated explicitly here so
that the resulting GLB should be evaluated against ``gaussians_final.ply`` and
not against an already axis-fixed prediction PLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_GT_TO_PREDICTION_ROTATION = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write a transformed copy of a reference GLB while preserving the "
            "original file"
        )
    )
    parser.add_argument("--input", required=True, help="Original reference GLB")
    parser.add_argument("--output", required=True, help="New transformed GLB")
    parser.add_argument(
        "--rotation",
        nargs=9,
        type=float,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        default=DEFAULT_GT_TO_PREDICTION_ROTATION.reshape(-1).tolist(),
        help=(
            "Row-major 3x3 GT-to-prediction rotation. The default is "
            "[-1 0 0; 0 0 1; 0 1 0]."
        ),
    )
    parser.add_argument(
        "--translation",
        nargs=3,
        type=float,
        metavar=("TX", "TY", "TZ"),
        default=(0.0, 0.0, 0.0),
        help="Optional translation applied after rotation; default is zero",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file; the input is still protected",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounds(scene) -> list[list[float]] | None:
    bounds = scene.bounds
    if bounds is None:
        return None
    return np.asarray(bounds, dtype=np.float64).tolist()


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if input_path.suffix.lower() != ".glb" or output_path.suffix.lower() != ".glb":
        raise ValueError("Both --input and --output must use the .glb extension")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input GLB does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Refusing to overwrite the input GLB; choose a new --output path")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite only if intended."
        )

    rotation = np.asarray(args.rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(args.translation, dtype=np.float64)
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-6 or not np.isclose(determinant, 1.0, atol=1.0e-6):
        raise ValueError(
            "--rotation must be a proper rigid rotation: "
            f"orthogonality_error={orthogonality_error:.6g}, det={determinant:.6g}"
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required; run pip install -r requirements.txt") from exc

    loaded = trimesh.load(input_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded
    if not scene.geometry:
        raise ValueError(f"Input GLB contains no mesh geometry: {input_path}")

    original_bounds = _bounds(scene)
    original_hash = _sha256(input_path)
    scene.apply_transform(transform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path, file_type="glb")

    verified = trimesh.load(output_path, force="scene", process=False)
    if isinstance(verified, trimesh.Trimesh):
        verified = trimesh.Scene(verified)
    if not verified.geometry:
        raise RuntimeError("Export verification failed: transformed GLB has no geometry")

    report = {
        "input_glb": str(input_path),
        "output_glb": str(output_path),
        "input_sha256_before": original_hash,
        "input_sha256_after": _sha256(input_path),
        "input_preserved": _sha256(input_path) == original_hash,
        "coordinate_mapping": "GT/CAD frame -> original Radar-GS prediction frame",
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "homogeneous_transform": transform.tolist(),
        "rotation_determinant": determinant,
        "rotation_orthogonality_error": orthogonality_error,
        "original_bounds": original_bounds,
        "transformed_bounds": _bounds(verified),
        "geometry_count": len(verified.geometry),
    }
    report_path = output_path.with_suffix(".transform.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("GT/CAD -> prediction transform:")
    print(transform)
    print(f"Original preserved: {report['input_preserved']} ({input_path})")
    print(f"Saved transformed GLB: {output_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
