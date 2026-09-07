#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc
base=outputs/radar_gs/adjusted_split_ablation_v3_20260825/asymmetric
root=outputs/radar_gs/k5_surface_smoothness_20260825/kernel_sweep
mkdir -p "$root"
views_file="$root/k5_fixed_views.json"
BASE_SELECTED="$base/selected_views.json" VIEWS_FILE="$views_file" /root/miniconda3/bin/python - <<'PY'
import json, os
d=json.load(open(os.environ['BASE_SELECTED']))
views=[]
for encounter in d['selection']:
    views.extend({'encounter':encounter,'frame_index':i} for i in (0,10,20,30,40,50))
json.dump({'seed':42,'maximum_views':len(views),'views':views},open(os.environ['VIEWS_FILE'],'w'),indent=2)
PY
run_one() {
  radius="$1"
  out="$root/radius_${radius}"
  mkdir -p "$out/checkpoints"
  SRC="$base/checkpoints/final_eval_raw_target.pt" DST="$out/checkpoints/final.pt" RADIUS="$radius" \
    /root/miniconda3/bin/python - <<'PY'
import os, torch
d=torch.load(os.environ['SRC'],map_location='cpu',weights_only=False)
d['arguments']['kernel_radius']=int(os.environ['RADIUS'])
torch.save(d,os.environ['DST'])
PY
  cp "$base/selected_views.json" "$out/selected_views.json"
  /root/miniconda3/bin/python -u evaluate_ablation.py \
    --checkpoint "$out/checkpoints/final.pt" \
    --data_root ../datasets/skynet5d --run_dir "$out" \
    --out_dir "$out/evaluation" --device cuda --split all \
    --evaluation_views_file "$views_file" \
    --preview_count 8 --q_thresholds 0.3 0.5 --no_export_fused_ply \
    > "$out/evaluation.log" 2>&1
}
export -f run_one
export base root views_file
printf '%s\n' 4 8 12 | xargs -P 2 -I '{}' bash -c 'run_one "$1"' _ '{}'
