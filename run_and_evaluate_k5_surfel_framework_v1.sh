#!/usr/bin/env bash
set -euo pipefail
cd /root/Yao/radar_gs_isar_enc
bash run_k5_surfel_framework_v1.sh
bash evaluate_k5_surfel_framework_v1.sh

