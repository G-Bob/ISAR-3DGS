#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

train_root=outputs/radar_gs/k1_random_20260827/training
while [[ $(find "$train_root" -mindepth 3 -maxdepth 3 \
  -path '*/checkpoints/final.pt' -type f | wc -l) -lt 2 ]]; do
  sleep 60
done
./evaluate_k1_random_2way.sh
