#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

train_root=outputs/radar_gs/k5_legacy_dynamics_20260826/training
eval_root=outputs/radar_gs/k5_legacy_dynamics_20260826/evaluation
mkdir -p "$eval_root"

prepare_and_eval() {
  local name="$1"
  local run="$train_root/$name"
  local out="$eval_root/$name"
  mkdir -p "$out/checkpoints"
  SRC="$run/checkpoints/final.pt" DST="$out/checkpoints/final.pt" \
    /root/miniconda3/bin/python - <<'PY'
import os
import torch

checkpoint = torch.load(os.environ["SRC"], map_location="cpu", weights_only=False)
checkpoint["arguments"]["background_threshold"] = 0.0
torch.save(checkpoint, os.environ["DST"])
PY
  cp "$run/selected_views.json" "$out/selected_views.json"
  /root/miniconda3/bin/python -u evaluate_ablation.py \
    --checkpoint "$out/checkpoints/final.pt" \
    --data_root ../datasets/skynet5d --run_dir "$out" --out_dir "$out/results" \
    --device cuda --split heldout --eval_encounter_count 44 \
    --max_eval_views 240 --eval_seed 42 --preview_count 8 \
    --q_thresholds 0.3 0.5 --no_export_fused_ply \
    --gt_geometry /root/Yao/3dmodel/skynet5d_half_scale_prediction_frame.glb \
    > "$out/evaluation.log" 2>&1
}
export -f prepare_and_eval
export train_root eval_root

printf '%s\n' legacy_schedule legacy_split legacy_prune legacy_thresholds legacy_full \
  | xargs -P 5 -I '{}' bash -c 'prepare_and_eval "$1"' _ '{}'
