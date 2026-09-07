#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc

train_root=outputs/radar_gs/k5_amplitude_psf_20260827/training
while [[ $(find "$train_root" -mindepth 3 -maxdepth 3 \
  -path '*/checkpoints/final.pt' -type f | wc -l) -lt 3 ]]; do
  sleep 60
done
./evaluate_k5_amplitude_psf_3way.sh
