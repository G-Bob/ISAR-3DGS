#!/usr/bin/env bash
ROOT=/rds/homes/g/gey/Yao/programs/GEIST-5/datasets/skynet5d_real_encounters
BASE=outputs/radar_vggt_k148_20260918
for k in 1 4 8; do
 name=k$k-radar_vggt; init=$BASE/$name/init_radar_vggt.npz; log=$BASE/logs/$name.log
 (python -u scripts/generate_radar_vggt_init.py --data_root "$ROOT" --out "$init" --encounter_count "$k" --images_per_encounter 50 --count 10000 --view_count 16 --patch_grid 16 --steps 120 && python -u run_reconstruction.py --config configs/full_backprojection.json --data_root "$ROOT" --images_per_encounter 50 --los_angle_deg 0 --num_gaussians 10000 --maximum_gaussians 20000 --iters 12000 --views_per_batch 4 --seed 42 --device cuda --encounter_count "$k" --initialization pointcloud_surface --init_path "$init" --out_dir "$BASE/$name") >"$log" 2>&1 &
 echo "$! $name" >> "$BASE/hybrid_pids.txt"
done
