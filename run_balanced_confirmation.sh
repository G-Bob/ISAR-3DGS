#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc
out="outputs/radar_gs/support_screen_20260825/combined_balanced"
/root/miniconda3/bin/python -u run_reconstruction.py \
  --config configs/full_backprojection.json \
  --data_root ../datasets/skynet5d \
  --out_dir "$out" \
  --encounter_preset_json ../datasets/skynet5d/preset_batch_20260822_5.json \
  --frames_per_encounter 60 --images_per_encounter 60 --los_angle_deg 0 \
  --background_threshold 0.10 \
  --num_gaussians 10000 --maximum_gaussians 20000 \
  --initial_support_probability 0.20 --initial_tangent_scale 0.04 \
  --lambda_support_sparsity 0.001 --lambda_support_binary 0.001 \
  --lambda_support_graph 0.01 --lambda_surface_gap 0 \
  --adaptive_interval 0 --split_mode none \
  --stage_a_fraction 0.0833333333 --stage_b_fraction 0.3333333333 \
  --stage_c_fraction 0.6666666667 \
  --iters 6000 --views_per_batch 8 --checkpoint_interval 6000 \
  --preview_interval 1000 --device cuda
