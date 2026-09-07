import json
from pathlib import Path

arms = [
    ("none", Path("outputs/radar_gs/adjusted_split_ablation_20260825/none")),
    ("asymmetric", Path("outputs/radar_gs/adjusted_split_ablation_v3_20260825/asymmetric")),
    ("symmetric", Path("outputs/radar_gs/adjusted_split_ablation_v3_20260825/symmetric")),
]

print("arm,psnr,ssim,mae,G,q03,outlier,bboxdiag,softvox,overlap,maxoverlap,pruned,split,t1,t2")
for arm, path in arms:
    evaluation = json.loads((path / "evaluation_raw_target/evaluation_metrics.json").read_text())
    history = json.loads((path / "training_history.json").read_text())
    image = evaluation["image_metrics"]
    geometry = evaluation["ply_metrics"]["thresholds"]["0.3"]
    volume = geometry["support_volume_summary"]
    adaptive = [row["adaptive_control"] for row in history if "adaptive_control" in row]
    values = [
        arm,
        image["mean_psnr_db"], image["mean_ssim"], image["mean_mae"],
        history[-1]["gaussians"], geometry["point_count"], geometry["knn_outlier_ratio"],
        geometry["bbox_diagonal_m"], volume["soft_support_voxel_count"],
        volume["multi_gaussian_overlap_fraction"], volume["maximum_overlap_count"],
        sum(item["pruned"] for item in adaptive), sum(item["split"] for item in adaptive),
        sum(item["split_axis_t1"] for item in adaptive),
        sum(item["split_axis_t2"] for item in adaptive),
    ]
    print(",".join(str(value) for value in values))
