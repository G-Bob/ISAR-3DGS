#!/usr/bin/env bash
set -euo pipefail

data_root=${1:?Usage: run_angle_ablation.sh DATA_ROOT OUTPUT_ROOT}
output_root=${2:?Usage: run_angle_ablation.sh DATA_ROOT OUTPUT_ROOT}

for angle in 0 2 5 10 15; do
  python run_reconstruction.py \
    --config configs/full_backprojection.json \
    --data_root "$data_root" \
    --out_dir "$output_root/los_${angle}deg" \
    --los_angle_deg "$angle"
done

