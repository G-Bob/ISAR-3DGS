#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

train_root=outputs/radar_gs/k5_pca_radius_refinement_20260826/training
expected=5

while true; do
  completed=$(find "$train_root" -mindepth 3 -maxdepth 3 \
    -path '*/checkpoints/final.pt' -type f | wc -l)
  if [[ "$completed" -eq "$expected" ]]; then
    break
  fi
  sleep 60
done

./evaluate_k5_pca_radius_refinement_5way.sh
