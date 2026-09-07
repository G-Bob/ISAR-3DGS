#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-fs/radar_gs_isar_enc/workspaces/v18_scattering_gate
v18=/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d2_k4_k10_scattering_gate_v18_20260902
log=/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/v19_queue.log

echo "waiting_for_v18 $(date -Is)"
while [[ ! -f "$v18/ALL_COMPLETE" ]]; do sleep 60; done
cd "$workspace"
echo "selecting_v18_weight $(date -Is)"
/root/miniconda3/bin/python select_v18_best_weight.py >> "$log" 2>&1
source "$v18/selected_weight.env"
echo "starting_k1_k10 selected=$SELECTED_RUN weight=$SCATTER_WEIGHT $(date -Is)"
bash run_k1_k10_optimal_v19_10way.sh "$SCATTER_WEIGHT"
echo "k1_k10_complete $(date -Is)"
