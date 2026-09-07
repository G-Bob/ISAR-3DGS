#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc
root=outputs/radar_gs/k5_surface_smoothness_20260825/evaluation
mkdir -p "$root"
base=outputs/radar_gs/adjusted_split_ablation_v3_20260825/asymmetric

prepare_and_eval() {
  local name="$1" run="$2" composition="$3"
  local out="$root/$name"
  mkdir -p "$out/checkpoints"
  SRC="$run/checkpoints/final.pt" DST="$out/checkpoints/final.pt" COMP="$composition" \
    /root/miniconda3/bin/python - <<'PY'
import os,torch
d=torch.load(os.environ['SRC'],map_location='cpu',weights_only=False)
d['arguments']['background_threshold']=0.0
d['arguments']['kernel_radius']=4
d['arguments']['support_composition']=os.environ['COMP']
torch.save(d,os.environ['DST'])
PY
  cp "$run/selected_views.json" "$out/selected_views.json"
  /root/miniconda3/bin/python -u evaluate_ablation.py \
    --checkpoint "$out/checkpoints/final.pt" --data_root ../datasets/skynet5d \
    --run_dir "$out" --out_dir "$out/results" --device cuda --split heldout \
    --eval_encounter_count 44 --max_eval_views 240 --eval_seed 42 \
    --preview_count 8 --q_thresholds 0.3 0.5 --no_export_fused_ply \
    --gt_geometry /root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.glb \
    > "$out/evaluation.log" 2>&1
}
export -f prepare_and_eval
export root base
printf '%s\n' \
  "baseline_noisy $base noisy_or" \
  "baseline_optical $base optical_density" \
  "surface_smooth outputs/radar_gs/k5_surface_smoothness_20260825/training_r4/surface_smooth optical_density" \
  "coverage_band outputs/radar_gs/k5_surface_smoothness_20260825/training_r4/coverage_band optical_density" \
  "conservative_split outputs/radar_gs/k5_surface_smoothness_20260825/training_r4/conservative_split optical_density" \
  | xargs -P 4 -n 3 bash -c 'prepare_and_eval "$@"' _
