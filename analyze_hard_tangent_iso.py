from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("outputs/radar_gs")
RUNS = {
    "support_mid": ROOT / "k5_schedule_prune_refine_20260826/evaluation/support_mid/results",
    "coverage_weight_010": ROOT / "k5_overlap_control_20260826/evaluation/coverage_weight_010/results",
    **{
        name: ROOT / "k5_hard_tangent_iso_20260827/evaluation" / name / "results"
        for name in (
            "iso_axis",
            "iso_local_pca",
            "iso_no_split",
            "iso_disp020",
            "iso_disp015",
            "iso_local_pca_disp020",
            "iso_local_pca_normal020",
        )
    },
}


def roughness(path: Path) -> tuple[float, float, float, float]:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    width = image.shape[1] // 5
    values = []
    for panel in (image[:, 3 * width : 4 * width], image[:, 4 * width : 5 * width]):
        values.append(
            float(
                0.5
                * (
                    np.abs(np.diff(panel, axis=0)).mean()
                    + np.abs(np.diff(panel, axis=1)).mean()
                )
            )
        )
        laplacian = (
            -4 * panel[1:-1, 1:-1]
            + panel[:-2, 1:-1]
            + panel[2:, 1:-1]
            + panel[1:-1, :-2]
            + panel[1:-1, 2:]
        )
        values.append(float(np.abs(laplacian).mean()))
    return tuple(values)


for name, run in RUNS.items():
    data = json.loads((run / "evaluation_metrics.json").read_text())
    image = data["image_metrics"]
    ply = data["ply_metrics"]
    threshold = ply["thresholds"]["0.3"]
    ground_truth = threshold["ground_truth"]
    voxel = threshold["voxel_metrics"]
    volume = threshold["support_volume_summary"]
    preview_values = [roughness(path) for path in sorted((run / "heldout_previews").glob("*.png"))]
    rough = np.asarray(preview_values).mean(axis=0)
    print(
        json.dumps(
            {
                "name": name,
                "psnr": image["mean_psnr_db"],
                "ssim": image["mean_ssim"],
                "mae": image["mean_mae"],
                "G": ply["original_point_count"],
                "q03": threshold["point_count"],
                "chamfer": ground_truth["chamfer_l1_m"],
                "hd95": ground_truth["hausdorff_p95_m"],
                "normal": ground_truth["normal_consistency_abs_cosine"],
                "iou": voxel["iou_3d"],
                "softvox": volume["soft_support_voxel_count"],
                "overlap": volume["multi_gaussian_overlap_fraction"],
                "maxov": volume["maximum_overlap_count"],
                "largest": threshold["largest_component_ratio"],
                "components": threshold["component_count"],
                "support_tv": rough[0],
                "support_lap": rough[1],
                "overlap_tv": rough[2],
                "overlap_lap": rough[3],
            },
            separators=(",", ":"),
        )
    )
