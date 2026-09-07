#!/usr/bin/env bash
set -euo pipefail
work=/root/autodl-fs/radar_gs_isar_enc/workspaces/v21_sequential_drift_20260907
output=/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d3_sequential_v21_20260907
cd "$work"
exec /root/miniconda3/bin/python -u live_dashboard.py \
  --output-root "$output" --host 127.0.0.1 --port 6006 --maximum-points 8000
