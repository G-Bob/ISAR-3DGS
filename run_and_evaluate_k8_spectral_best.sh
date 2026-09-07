#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

root=outputs/radar_gs/k8_spectral_best_20260826
train="$root/training"
eval_dir="$root/evaluation"
mkdir -p "$train"

/root/miniconda3/bin/python -u run_reconstruction.py \
  --config configs/full_backprojection.json \
  --data_root ../datasets/skynet5d --out_dir "$train" \
  --encounter_preset_json ../datasets/skynet5d/preset_batch_20260822_8.json \
  --frames_per_encounter 60 --images_per_encounter 60 --los_angle_deg 0 \
  --num_gaussians 10000 --maximum_gaussians 20000 \
  --background_threshold 0.1 \
  --initial_tangent_scale 0.04 --initial_normal_scale 0.012 \
  --initial_support_probability 0.2 \
  --lambda_support_sparsity 0.003 --lambda_support_binary 0.001 \
  --lambda_support_graph 0.005 --lambda_surface_gap 0 \
  --lambda_bidirectional_plane 0.2 --lambda_scale_smoothness 0.05 \
  --lambda_coverage_band 0.05 --coverage_min_ratio 0.75 \
  --coverage_max_ratio 1.25 \
  --surface_graph_mode spatial --lambda_reference_normal 0.05 \
  --pca_normal_refresh_mode adaptive_hard --pca_min_confidence 0.0001 \
  --pca_max_neighbor_distance 0.0 \
  --pca_max_surface_variation 0.05 \
  --pca_min_planar_spread_ratio 0.15 \
  --kernel_radius 4 --support_composition optical_density \
  --split_support_threshold 0.5 --split_gradient_threshold 0.00005 \
  --split_evidence_threshold 0.0001 --adaptive_interval 500 \
  --split_displacement_ratio 0.3 --split_scale_shrink 0.8 \
  --split_mode asymmetric --split_cooldown_iterations 1000 \
  --stage_a_fraction 0.0833333333 --stage_b_fraction 0.3333333333 \
  --stage_c_fraction 0.6666666667 \
  --iters 12000 --views_per_batch 8 --checkpoint_interval 500 \
  --preview_interval 500 --device cuda \
  > "$train/train.log" 2>&1

mkdir -p "$eval_dir/checkpoints"
SRC="$train/checkpoints/final.pt" DST="$eval_dir/checkpoints/final.pt" \
  /root/miniconda3/bin/python - <<'PY'
import os
import torch

checkpoint = torch.load(os.environ["SRC"], map_location="cpu", weights_only=False)
checkpoint["arguments"]["background_threshold"] = 0.0
torch.save(checkpoint, os.environ["DST"])
PY
cp "$train/selected_views.json" "$eval_dir/selected_views.json"

/root/miniconda3/bin/python -u evaluate_ablation.py \
  --checkpoint "$eval_dir/checkpoints/final.pt" \
  --data_root ../datasets/skynet5d --run_dir "$eval_dir" \
  --out_dir "$eval_dir/results" --device cuda --split heldout \
  --eval_encounter_count 44 --max_eval_views 240 --eval_seed 42 \
  --preview_count 8 --q_thresholds 0.3 0.5 --no_export_fused_ply \
  --gt_geometry /root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.glb \
  > "$eval_dir/evaluation.log" 2>&1
