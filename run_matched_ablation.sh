#!/usr/bin/env bash
set -euo pipefail

cd /root/Yao/radar_gs_isar_enc
mkdir -p outputs/radar_gs/ablation_split_matched_20260825/logs

# Reuse the already-running symmetric arm, then run the two remaining arms
# sequentially to avoid GPU contention and non-deterministic timing effects.
while kill -0 55692 2>/dev/null; do
  sleep 30
done

run_arm() {
  local mode="$1"
  local out_dir="outputs/radar_gs/ablation_split_matched_20260825/${mode}"
  /root/miniconda3/bin/python -u run_reconstruction.py \
    --config configs/full_backprojection.json \
    --data_root ../datasets/skynet5d \
    --out_dir "$out_dir" \
    --encounter_preset_json ../datasets/skynet5d/preset_batch_20260822_5.json \
    --frames_per_encounter 60 \
    --images_per_encounter 60 \
    --los_angle_deg 0 \
    --num_gaussians 10000 \
    --maximum_gaussians 20000 \
    --initial_tangent_scale 0.08 \
    --lambda_support_sparsity 0.001 \
    --lambda_support_binary 0.001 \
    --lambda_support_graph 0.01 \
    --lambda_surface_gap 0 \
    --split_support_threshold 0.6 \
    --split_gradient_threshold 0.0001 \
    --adaptive_interval 1000 \
    --split_displacement_ratio 0.3 \
    --split_scale_shrink 0.8 \
    --split_mode "$mode" \
    --iters 40000 \
    --views_per_batch 8 \
    --device cuda \
    2>&1 | tee "outputs/radar_gs/ablation_split_matched_20260825/logs/${mode}.log"
}

run_arm none
run_arm asymmetric
