#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

run_eval() {
  local dir="$1"
  local checkpoint="$dir/checkpoints/final_eval_raw_target.pt"
  if [[ ! -f "$checkpoint" ]]; then
    DIR="$dir" /root/miniconda3/bin/python -c 'import os, torch, pathlib; p=pathlib.Path(os.environ["DIR"])/"checkpoints/final.pt"; q=p.with_name("final_eval_raw_target.pt"); d=torch.load(p,map_location="cpu",weights_only=False); d["arguments"]["background_threshold"]=0.0; torch.save(d,q)'
  fi
  /root/miniconda3/bin/python -u evaluate_ablation.py \
    --checkpoint "$checkpoint" --data_root ../datasets/skynet5d \
    --run_dir "$dir" --out_dir "$dir/evaluation_leave_own_out_240" \
    --device cuda --split heldout --eval_encounter_count 44 \
    --max_eval_views 240 --eval_seed 42 --preview_count 0 \
    --q_thresholds 0.3 0.5 --no_export_fused_ply \
    > "$dir/evaluation_leave_own_out_240.log" 2>&1
}
export -f run_eval

printf '%s\n' \
  outputs/radar_gs/asymmetric_k2_k4_parallel_20260825/k03 \
  outputs/radar_gs/asymmetric_k2_k4_parallel_20260825/k04 \
  outputs/radar_gs/adjusted_split_ablation_v3_20260825/asymmetric \
  outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k06 \
  outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k07 \
  outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k08 \
  outputs/radar_gs/asymmetric_k6_k9_parallel_20260825/k09 \
  | xargs -P 3 -I '{}' bash -c 'run_eval "$1"' _ '{}'
