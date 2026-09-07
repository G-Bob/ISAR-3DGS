#!/usr/bin/env bash
set -euo pipefail

cd /root/Yao/radar_gs_isar_enc
root_out="outputs/radar_gs/support_screen_20260825"
mkdir -p "$root_out/logs"

run_arm() {
  local name="$1"
  local threshold="$2"
  local support="$3"
  local tangent="$4"
  local sparsity="$5"
  local graph="$6"
  /root/miniconda3/bin/python -u run_reconstruction.py \
    --config configs/full_backprojection.json \
    --data_root ../datasets/skynet5d \
    --out_dir "$root_out/$name" \
    --encounter_preset_json ../datasets/skynet5d/preset_batch_20260822_5.json \
    --frames_per_encounter 60 \
    --images_per_encounter 60 \
    --los_angle_deg 0 \
    --background_threshold "$threshold" \
    --num_gaussians 10000 \
    --maximum_gaussians 20000 \
    --initial_support_probability "$support" \
    --initial_tangent_scale "$tangent" \
    --lambda_support_sparsity "$sparsity" \
    --lambda_support_binary 0.001 \
    --lambda_support_graph "$graph" \
    --lambda_surface_gap 0 \
    --adaptive_interval 0 \
    --split_mode none \
    --stage_a_fraction 0.0833333333 \
    --stage_b_fraction 0.3333333333 \
    --stage_c_fraction 0.6666666667 \
    --iters 6000 \
    --views_per_batch 8 \
    --checkpoint_interval 6000 \
    --preview_interval 1000 \
    --device cuda \
    2>&1 | tee "$root_out/logs/$name.log"
}

# Diagnose the three most plausible causes before another long run.
run_arm filtered_only 0.10 0.70 0.08 0.001 0.01
run_arm compact_support_only 0.00 0.20 0.04 0.001 0.01
run_arm combined_sparse 0.10 0.20 0.04 0.003 0.005
