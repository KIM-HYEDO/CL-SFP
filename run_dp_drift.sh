#!/usr/bin/env bash
# Diffusion Policy drift curves on a 250-step robomimic task, three seeds:
# train + static sweep if missing (run_dp_3seeds.sh, which skips what exists),
# then the drift curve at each seed's selected checkpoint on the same nine
# levels fair_eval.sh uses for SFP on can/square, 100 eval seeds.
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_dp_drift.sh can
set -e
cd /workspace/CL-SFP; mkdir -p logs
TASK=$1; [ -n "$TASK" ] || { echo "usage: $0 <task>"; exit 1; }
LEVELS="0.0,0.0001,0.00025,0.0003,0.0005,0.00075,0.001,0.00125,0.0015"
bash run_dp_3seeds.sh $TASK
echo "=== $TASK dp drift curves ==="
pids=()
for SUF in "" _seed1 _seed2; do
    bash drift_eval.sh $TASK dp$SUF $LEVELS 100 > logs/dp_drift_${TASK}$SUF.log 2>&1 & pids+=($!); sleep 5
done
for p in "${pids[@]}"; do wait $p || echo "DRIFT FAILED (pid $p)"; done
echo "=== $TASK DP_DRIFT_DONE ==="
