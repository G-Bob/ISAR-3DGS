#!/usr/bin/env bash
set -euo pipefail

cd /root/Yao/radar_gs_isar_enc
root_out="outputs/radar_gs/asymmetric_k2_k4_parallel_20260825"
mkdir -p "$root_out/logs"

run_k() {
  local k="$1"
  local label
  label=$(printf "k%02d" "$k")
  /root/miniconda3/bin/python -u run_reconstruction.py \
    --config configs/full_backprojection.json --data_root ../datasets/skynet5d \
    --out_dir "$root_out/$label" \
    --encounter_preset_json "../datasets/skynet5d/preset_batch_20260822_${k}.json" \
    --frames_per_encounter 60 --images_per_encounter 60 --los_angle_deg 0 \
    --background_threshold 0.10 --num_gaussians 10000 --maximum_gaussians 20000 \
    --initial_support_probability 0.20 --initial_tangent_scale 0.04 \
    --lambda_support_sparsity 0.003 --lambda_support_binary 0.001 \
    --lambda_support_graph 0.005 --lambda_surface_gap 0 \
    --prune_support_threshold 0.05 --prune_evidence_threshold 0.0001 \
    --prune_min_visible_fraction 0.30 --split_support_threshold 0.50 \
    --split_gradient_threshold 0.00005 --split_evidence_threshold 0.0001 \
    --split_displacement_ratio 0.30 --split_scale_shrink 0.80 \
    --split_mode asymmetric --adaptive_interval 500 \
    --stage_a_fraction 0.0833333333 --stage_b_fraction 0.3333333333 \
    --stage_c_fraction 0.6666666667 --iters 12000 --views_per_batch 8 \
    --checkpoint_interval 4000 --preview_interval 1000 --device cuda \
    > "$root_out/logs/$label.log" 2>&1
}

pids=()
for k in 2 3 4; do
  run_k "$k" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
