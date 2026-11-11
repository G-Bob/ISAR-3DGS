#!/bin/bash
cd /rds/homes/g/gey/Yao/programs/ISAR-3DGS
while pgrep -f "q_robust_20261018/robust_suite.py" > /dev/null; do sleep 60; done
echo "phase1 local done $(date)"
python -u outputs/q_robust2_20261018/robust_suite.py
