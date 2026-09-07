#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

root=outputs/radar_gs/k5_input_decode_20260828/training
mkdir -p "$root"

run_variant() {
  local name="$1" processing_mode="$2"
  local out="$root/$name"
  mkdir -p "$out"
  /root/miniconda3/bin/python -u run_reconstruction.py \
    --config configs/full_backprojection.json \
    --data_root ../datasets/skynet5d --out_dir "$out" \
    --encounter_preset_json ../datasets/skynet5d/preset_batch_20260822_5.json \
    --frames_per_encounter 60 --images_per_encounter 60 --los_angle_deg 0 \
    --amplitude_processing_mode "$processing_mode" \
    --legacy_background_threshold_for_decode 0.1 \
    --num_gaussians 10000 --maximum_gaussians 20000 \
    --background_threshold 0.1 \
    --initial_tangent_scale 0.04 --initial_normal_scale 0.012 \
    --tangent_covariance_mode isotropic_geometric \
    --normal_covariance_floor_ratio 0.25 \
    --initial_support_probability 0.2 \
    --lambda_support_sparsity 0.003 --lambda_support_binary 0.001 \
    --lambda_support_graph 0.005 --lambda_surface_gap 0 \
    --lambda_bidirectional_plane 0.2 --lambda_scale_smoothness 0.05 \
    --lambda_tangent_isotropy 0 --lambda_covariance_consistency 0 \
    --lambda_normal_consistency 0.10 \
    --lambda_coverage_band 0.10 --coverage_min_ratio 0.75 \
    --coverage_max_ratio 1.25 --maximum_normal_ratio 0.20 \
    --surface_graph_mode spatial --lambda_reference_normal 0.05 \
    --pca_normal_refresh_mode adaptive_hard --pca_min_confidence 0.0001 \
    --pca_max_neighbor_distance 0.0 \
    --pca_max_surface_variation 0.05 \
    --pca_min_planar_spread_ratio 0.15 \
    --kernel_radius 4 --support_composition optical_density \
    --prune_support_threshold 0.075 --prune_min_visible_fraction 0.30 \
    --split_support_threshold 0.5 --split_gradient_threshold 0.00005 \
    --split_evidence_threshold 0.0001 --adaptive_interval 500 \
    --split_displacement_ratio 0.20 --split_scale_shrink 0.8 \
    --split_mode asymmetric --split_direction_mode tangent_axes \
    --split_cooldown_iterations 1000 \
    --stage_a_fraction 0.15 --stage_b_fraction 0.50 \
    --stage_c_fraction 0.80 \
    --iters 12000 --views_per_batch 8 --checkpoint_interval 500 \
    --preview_interval 500 --seed 42 --device cuda \
    > "$out/train.log" 2>&1
}
export -f run_variant
export root

printf '%s\n' \
  'legacy_rgb_max legacy_rgb_max' \
  'inverse_hot_db inverse_hot_db' \
  'inverse_hot_linear inverse_hot_linear' \
  | xargs -P 3 -n 2 bash -c 'run_variant "$@"' _
