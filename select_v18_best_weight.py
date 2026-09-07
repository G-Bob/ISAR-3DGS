import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/"
    "skynet5d2_k4_k10_scattering_gate_v18_20260902/evaluation"
)
WEIGHTS = {
    "k4_scatter0000": 0.0,
    "k4_scatter0001": 0.001,
    "k4_scatter0025": 0.0025,
    "k4_scatter0050": 0.005,
    "k4_scatter0075": 0.0075,
    "k4_scatter0100": 0.01,
}


def load(name: str) -> dict:
    records = []
    for test in ("k5_test3", "k9_test7"):
        path = ROOT / f"{name}_on_{test}" / "evaluation_metrics.json"
        data = json.loads(path.read_text())
        image = data["image_metrics"]
        geom = data["ply_metrics"]["thresholds"]["0.3"]
        records.append((image, geom))
    image0, geom = records[0]
    gt = geom["ground_truth"]
    voxel = geom["voxel_metrics"]
    return {
        "name": name,
        "weight": WEIGHTS[name],
        "psnr": sum(r[0]["mean_psnr_db"] for r in records) / 2,
        "ssim": sum(r[0]["mean_ssim"] for r in records) / 2,
        "chamfer": gt["chamfer_l1_m"],
        "hd95": gt["hausdorff_p95_m"],
        "iou": voxel["iou_3d"],
        "overlap": geom["support_volume_summary"]["multi_gaussian_overlap_fraction"],
        "largest_component": geom["largest_component_ratio"],
    }


rows = [load(name) for name in WEIGHTS]
best_psnr = max(row["psnr"] for row in rows)
best_hd95 = min(row["hd95"] for row in rows)
# Reject solutions that buy IoU by visibly degrading image fidelity or tail geometry.
candidates = [
    row for row in rows
    if row["psnr"] >= best_psnr - 0.15 and row["hd95"] <= best_hd95 * 1.05
]
if not candidates:
    raise RuntimeError("No v18 candidate satisfies the predeclared fidelity constraints")
selected = max(candidates, key=lambda row: (row["iou"], -row["chamfer"], -row["overlap"]))
report = {"selection_rule": "PSNR within 0.15 dB and HD95 within 5% of best; maximize IoU", "rows": rows, "selected": selected}
out = ROOT.parent / "selected_weight.json"
out.write_text(json.dumps(report, indent=2))
env = ROOT.parent / "selected_weight.env"
env.write_text(f"SELECTED_RUN={selected['name']}\nSCATTER_WEIGHT={selected['weight']}\n")
print(json.dumps(report, indent=2))
