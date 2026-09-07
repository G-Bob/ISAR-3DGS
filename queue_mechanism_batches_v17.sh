#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-fs/radar_gs_isar_enc/workspaces/v17_mechanism_batches
v16=/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d2_k4_multiencounter_prune_v16_20260902/ALL_COMPLETE
group1=/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d2_k4_mechanism_group1_v17_20260902/ALL_COMPLETE

echo "waiting_for_v16 $(date -Is)"
while [[ ! -f "$v16" ]]; do sleep 60; done
echo "starting_group1 $(date -Is)"
cd "$workspace"
bash run_k4_mechanism_group1_v17_6way.sh
test -f "$group1"
echo "starting_group2 $(date -Is)"
bash run_k4_mechanism_group2_v17_6way.sh
echo "all_batches_complete $(date -Is)"
