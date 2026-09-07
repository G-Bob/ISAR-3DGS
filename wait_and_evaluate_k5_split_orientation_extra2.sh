#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

train_root=outputs/radar_gs/k5_split_orientation_20260827/training
while [[ ! -f "$train_root/tangent_iso_005/checkpoints/final.pt" \
       || ! -f "$train_root/local_pca_iso_005/checkpoints/final.pt" ]]; do
  sleep 60
done
./evaluate_k5_split_orientation_extra2.sh
