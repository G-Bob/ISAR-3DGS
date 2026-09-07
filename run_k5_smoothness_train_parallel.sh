#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc
root=outputs/radar_gs/k5_surface_smoothness_20260825/training_r4
mkdir -p "$root"

run_variant() {
  local name="$1" bidir="$2" scale="$3" coverage="$4" split_mode="$5" disp="$6" shrink="$7"
  local out="$root/$name"
  mkdir -p "$out"
  /root/miniconda3/bin/python -u run_reconstruction.py \
    --config configs/full_backprojection.json \
    --data_root ../datasets/skynet5d --out_dir "$out" \
    --encounter_preset_json ../datasets/skynet5d/preset_batch_20260822_5.json \
    --frames_per_encounter 60 --images_per_encounter 60 --los_angle_deg 0 \
    --num_gaussians 10000 --maximum_gaussians 20000 \
    --background_threshold 0.1 \
    --initial_tangent_scale 0.04 --initial_normal_scale 0.012 \
    --initial_support_probability 0.2 \
    --lambda_support_sparsity 0.003 --lambda_support_binary 0.001 \
    --lambda_support_graph 0.005 --lambda_surface_gap 0 \
    --lambda_bidirectional_plane "$bidir" \
    --lambda_scale_smoothness "$scale" \
    --lambda_coverage_band "$coverage" \
    --coverage_min_ratio 0.75 --coverage_max_ratio 1.25 \
    --kernel_radius 4 --support_composition optical_density \
    --split_support_threshold 0.5 --split_gradient_threshold 0.00005 \
    --split_evidence_threshold 0.0001 --adaptive_interval 500 \
    --split_displacement_ratio "$disp" --split_scale_shrink "$shrink" \
    --split_mode "$split_mode" \
    --stage_a_fraction 0.0833333333 --stage_b_fraction 0.3333333333 \
    --stage_c_fraction 0.6666666667 \
    --iters 12000 --views_per_batch 8 --checkpoint_interval 4000 \
    --preview_interval 1000 --device cuda \
    > "$out/train.log" 2>&1
}
export -f run_variant
export root

printf '%s\n' \
  'surface_smooth 0.2 0.05 0 asymmetric 0.3 0.8' \
  'coverage_band 0.2 0.05 0.05 asymmetric 0.3 0.8' \
  'conservative_split 0.2 0.05 0.05 symmetric_conservative 0.7 0.65' \
  | xargs -P 3 -n 7 bash -c 'run_variant "$@"' _
