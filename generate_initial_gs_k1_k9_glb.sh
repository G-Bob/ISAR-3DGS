#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

root=outputs/radar_gs/initial_gs_k1_k9_glb_20260827
model=/root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.glb
mkdir -p "$root/logs"

generate_one() {
  local k="$1"
  local points="$2"
  local output="$root/K${k}_initial_gs_distance.glb"
  /root/miniconda3/bin/python -u scripts/overlay_gaussians_on_glb_distance_v2.py \
    --model "$model" \
    --points "$points" \
    --output "$output" \
    --alignment none \
    --qi-threshold 0.3 \
    --model-alpha 0.18 \
    --point-radius 0.02 \
    --point-subdivisions 1 \
    --color-mode distance \
    --near-distance 0.05 \
    --far-distance 0.50 \
    --outlier-nb-neighbors 0 \
    --surface-samples 100000 \
    --random-seed 7 \
    --alignment-report "$root/K${k}_initial_gs_distance.json" \
    > "$root/logs/K${k}.log" 2>&1
}
export -f generate_one
export root model

printf '%s\n' \
  '01 outputs/radar_gs/k1_random_20260827/evaluation/random_a/gaussians_final.ply' \
  '02 outputs/radar_gs/asymmetric_k2_k4_parallel_20260825/k02/gaussians_final.ply' \
  '03 outputs/radar_gs/asymmetric_k2_k4_parallel_20260825/k03/gaussians_final.ply' \
  '04 outputs/radar_gs/asymmetric_k2_k4_parallel_20260825/k04/gaussians_final.ply' \
  '05 outputs/radar_gs/adjusted_split_ablation_v3_20260825/asymmetric/gaussians_final.ply' \
  '06 outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k06/gaussians_final.ply' \
  '07 outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k07/gaussians_final.ply' \
  '08 outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k08/gaussians_final.ply' \
  '09 outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k09/gaussians_final.ply' \
  | xargs -P 3 -n 2 bash -c 'generate_one "$@"' _
