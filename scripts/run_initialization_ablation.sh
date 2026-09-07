#!/usr/bin/env bash
set -euo pipefail

data_root=${1:?Usage: run_initialization_ablation.sh DATA_ROOT OUTPUT_ROOT [INIT_PATH]}
output_root=${2:?Usage: run_initialization_ablation.sh DATA_ROOT OUTPUT_ROOT [INIT_PATH]}
init_path=${3:-}

for initialization in random_volume sphere_surface box_surface backprojection_surface; do
  python run_reconstruction.py \
    --config configs/init_ablation.json \
    --data_root "$data_root" \
    --out_dir "$output_root/$initialization" \
    --initialization "$initialization"
done

if [[ -n "$init_path" ]]; then
  case "${init_path##*.}" in
    obj) initialization=mesh_surface ;;
    *) initialization=pointcloud_surface ;;
  esac
  python run_reconstruction.py \
    --config configs/init_ablation.json \
    --data_root "$data_root" \
    --out_dir "$output_root/$initialization" \
    --initialization "$initialization" \
    --init_path "$init_path"
fi

