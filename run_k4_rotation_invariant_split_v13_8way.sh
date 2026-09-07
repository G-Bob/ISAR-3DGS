#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_root=/root/Yao/datasets/skynet5d-2
preset="$data_root/preset_batch_20260827_221116.json"
root=/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d2_k4_rotation_invariant_split_v13_20260831
mkdir -p "$root/training" "$root/evaluation"

run_variant() {
  local name="$1" split_mode="$2" direction_mode="$3" evidence_threshold="$4"
  local lambda_repulsion="$5" repulsion_ratio="$6" knn_scale_ratio="$7"
  local out="$root/training/$name"
  [[ -e "$out/train.log" ]] && { echo "REFUSE_OVERWRITE $out" >&2; return 2; }
  mkdir -p "$out"
  /root/miniconda3/bin/python -u run_reconstruction.py \
    --config configs/full_backprojection.json \
    --data_root "$data_root" --out_dir "$out" --encounter_preset_json "$preset" \
    --frames_per_encounter 60 --images_per_encounter 60 --los_angle_deg 0 \
    --num_gaussians 10000 --maximum_gaussians 20000 \
    --background_threshold 0.1 --amplitude_processing_mode control \
    --initial_tangent_scale 0.04 --initial_normal_scale 0.012 \
    --tangent_covariance_mode isotropic_geometric --normal_covariance_floor_ratio 0.25 \
    --primitive_mode gaussian3d --surface_field_mode offset_sdf \
    --surface_field_hidden_dim 32 --surface_field_layers 3 \
    --surface_field_start_fraction 0.15 --surface_field_full_fraction 0.55 \
    --lambda_surface_field 0.01 --lambda_surface_field_offset 1.0 \
    --lambda_surface_field_normal 0.002 --lambda_surface_field_geometry_normal 0 \
    --lambda_surface_field_eikonal 0.01 \
    --surface_field_minimum_support 0.30 --surface_field_minimum_evidence 0.0001 \
    --surface_field_sample_count 1024 --surface_field_eikonal_jitter 0.08 \
    --surface_field_offset_distance 0.05 --surface_field_offset_loss_mode normalized_l1 \
    --surface_field_projection_mode none \
    --adaptive_gradient_source blended --adaptive_surface_field_gradient_weight 0.50 \
    --initial_support_probability 0.2 \
    --lambda_support_sparsity 0.003 --lambda_support_binary 0.001 \
    --lambda_support_graph 0.005 --lambda_surface_gap 0 \
    --lambda_bidirectional_plane 0.2 --lambda_scale_smoothness 0.05 \
    --lambda_tangent_isotropy 0 --lambda_covariance_consistency 0 \
    --lambda_tangent_repulsion "$lambda_repulsion" \
    --tangent_repulsion_min_ratio "$repulsion_ratio" \
    --lambda_normal_consistency 0.10 --lambda_coverage_band 0.10 \
    --coverage_min_ratio 0.75 --coverage_max_ratio 1.25 --maximum_normal_ratio 0.20 \
    --surface_graph_mode spatial --lambda_reference_normal 0.05 \
    --pca_normal_refresh_mode adaptive_hard --pca_min_confidence 0.0001 \
    --pca_max_neighbor_distance 0.0 --pca_max_surface_variation 0.05 \
    --pca_min_planar_spread_ratio 0.15 \
    --kernel_radius 4 --support_composition optical_density \
    --prune_support_threshold 0.075 --prune_min_visible_fraction 0.30 \
    --split_support_threshold 0.5 --split_gradient_threshold 0.00005 \
    --split_evidence_threshold "$evidence_threshold" --adaptive_interval 500 \
    --split_displacement_ratio 0.20 --split_scale_shrink 0.80 \
    --split_mode "$split_mode" --split_direction_mode "$direction_mode" \
    --adaptive_knn_tangent_scale_ratio "$knn_scale_ratio" \
    --split_cooldown_iterations 1000 \
    --observation_aware_ema --observation_weighted_support \
    --observation_support_weight_floor 0.05 --observation_fov_margin_pixels 2 \
    --stage_a_fraction 0.15 --stage_b_fraction 0.50 --stage_c_fraction 0.80 \
    --iters 12000 --views_per_batch 8 --checkpoint_interval 500 \
    --preview_interval 500 --seed 42 --device cuda \
    > "$out/train.log" 2>&1
}
export -f run_variant
export data_root preset root

printf '%s\n' \
  'control asymmetric tangent_axes 0.0001 0 0.5 0' \
  'no_split none tangent_axes 0.0001 0 0.5 0' \
  'asym_center_grad asymmetric center_gradient 0.0001 0 0.5 0' \
  'sym_center_grad symmetric center_gradient 0.0001 0 0.5 0' \
  'sym_local_pca symmetric local_pca 0.0001 0 0.5 0' \
  'sym_grad_evidence5 symmetric center_gradient 0.0005 0 0.5 0' \
  'sym_grad_repel symmetric center_gradient 0.0001 0.02 0.75 0' \
  'sym_grad_knn080 symmetric center_gradient 0.0001 0 0.5 0.80' \
  | xargs -P 8 -n 7 bash -c 'run_variant "$@"' _

evaluate_variant() {
  local name="$1" test_name="$2" test_preset="$3"
  local run="$root/training/$name" out="$root/evaluation/${name}_on_${test_name}"
  mkdir -p "$out"
  /root/miniconda3/bin/python -u evaluate_ablation.py \
    --checkpoint "$run/checkpoints/final.pt" --data_root "$data_root" \
    --run_dir "$run" --out_dir "$out" --evaluation_preset_json "$data_root/$test_preset" \
    --device cuda --split all --max_eval_views 0 --eval_seed 42 \
    --preview_count 12 --q_thresholds 0.3 0.5 --no_export_fused_ply \
    --gt_geometry /root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.glb \
    > "$out/evaluation.log" 2>&1
}
export -f evaluate_variant
jobs_file="$root/final_evaluation_jobs.txt"; : > "$jobs_file"
for name in control no_split asym_center_grad sym_center_grad sym_local_pca sym_grad_evidence5 sym_grad_repel sym_grad_knn080; do
  printf '%s %s %s\n' "$name" k5_test3 preset_batch_20260828_011104_testset.json >> "$jobs_file"
  printf '%s %s %s\n' "$name" k9_test7 preset_batch_20260827_235101_testset.json >> "$jobs_file"
done
xargs -P 2 -n 3 bash -c 'evaluate_variant "$@"' _ < "$jobs_file"
touch "$root/ALL_COMPLETE"
