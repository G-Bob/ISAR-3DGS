#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

root=outputs/radar_gs/k5_legacy_dynamics_20260826/training
mkdir -p "$root"

run_variant() {
  local name="$1"
  local stage_a="$2"
  local stage_b="$3"
  local stage_c="$4"
  local split_support="$5"
  local split_gradient="$6"
  local prune_support="$7"
  local prune_visible="$8"
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
    --lambda_bidirectional_plane 0.2 --lambda_scale_smoothness 0.05 \
    --lambda_coverage_band 0.05 --coverage_min_ratio 0.75 \
    --coverage_max_ratio 1.25 \
    --surface_graph_mode spatial --lambda_reference_normal 0.05 \
    --pca_normal_refresh_mode adaptive_hard --pca_min_confidence 0.0001 \
    --pca_max_neighbor_distance 0.0 \
    --pca_max_surface_variation 0.05 \
    --pca_min_planar_spread_ratio 0.15 \
    --kernel_radius 4 --support_composition optical_density \
    --prune_support_threshold "$prune_support" \
    --prune_min_visible_fraction "$prune_visible" \
    --split_support_threshold "$split_support" \
    --split_gradient_threshold "$split_gradient" \
    --split_evidence_threshold 0.0001 --adaptive_interval 500 \
    --split_displacement_ratio 0.3 --split_scale_shrink 0.8 \
    --split_mode asymmetric --split_cooldown_iterations 1000 \
    --stage_a_fraction "$stage_a" --stage_b_fraction "$stage_b" \
    --stage_c_fraction "$stage_c" \
    --iters 12000 --views_per_batch 8 --checkpoint_interval 500 \
    --preview_interval 500 --device cuda \
    > "$out/train.log" 2>&1
}
export -f run_variant
export root

# Current values: stages 1/12,1/3,2/3; split 0.50/5e-5; prune 0.05/0.30.
# Legacy values: stages 0.15/0.50/0.80; split 0.60/1e-4; prune 0.10/0.20.
printf '%s\n' \
  'legacy_schedule 0.15 0.50 0.80 0.50 0.00005 0.05 0.30' \
  'legacy_split 0.0833333333 0.3333333333 0.6666666667 0.60 0.0001 0.05 0.30' \
  'legacy_prune 0.0833333333 0.3333333333 0.6666666667 0.50 0.00005 0.10 0.20' \
  'legacy_thresholds 0.0833333333 0.3333333333 0.6666666667 0.60 0.0001 0.10 0.20' \
  'legacy_full 0.15 0.50 0.80 0.60 0.0001 0.10 0.20' \
  | xargs -P 5 -n 8 bash -c 'run_variant "$@"' _
