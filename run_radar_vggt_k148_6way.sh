#!/usr/bin/env bash
set -u
ROOT=/rds/homes/g/gey/Yao/programs/GEIST-5/datasets/skynet5d_real_encounters
BASE=outputs/radar_vggt_k148_20260918
mkdir -p "$BASE/logs"; : > "$BASE/pids.txt"
for k in 1 4 8; do
  for init in backprojection_surface radar_vggt_surface; do
    short=$(echo "$init" | sed 's/_surface//'); name=k$k-$short; out="$BASE/$name"; log="$BASE/logs/$name.log"
    extra=""; if [ "$init" = radar_vggt_surface ]; then extra="--radar_vggt_view_count 16 --radar_vggt_patch_grid 16 --radar_vggt_steps 120 --radar_vggt_learning_rate 0.002"; fi
    nohup python -u run_reconstruction.py --config configs/full_backprojection.json --data_root "$ROOT" --images_per_encounter 50 --los_angle_deg 0 --num_gaussians 10000 --maximum_gaussians 20000 --iters 12000 --views_per_batch 4 --seed 42 --device cuda --encounter_count "$k" --initialization "$init" $extra --out_dir "$out" >"$log" 2>&1 &
    echo "$! $name" >> "$BASE/pids.txt"
  done
done
