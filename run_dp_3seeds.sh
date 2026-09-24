#!/usr/bin/env bash
# Diffusion Policy (DP-C) under this project's protocol, three seeds: train
# 1000 epochs with the shared recipe, then the static checkpoint sweep with
# 100 DDPM steps per chunk. See algo/dp.py for what is held fixed.
#
# Cost: DP inference is 100 network calls per 8 env steps, so an episode is
# ~5x an SFP episode (square 17.5 s, tool_hang ~55 s). The sweep dominates.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_dp_3seeds.sh square
set -e
cd /workspace/CL-SFP; mkdir -p logs
TASK=$1; [ -n "$TASK" ] || { echo "usage: $0 <task>"; exit 1; }
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
for SEED in 0 1 2; do
    SUF=""; [ $SEED -gt 0 ] && SUF="_seed$SEED"
    if [ -f outputs/$TASK/dp$SUF/ep1000.ckpt ]; then echo "skip train $TASK/dp$SUF"; continue; fi
    $PY algo/dp.py --task $TASK --mode train --epochs 1000 --train-seed $SEED --tag "$SUF" \
        > logs/dp_train_${TASK}$SUF.log 2>&1 && echo "trained $TASK/dp$SUF" || echo "TRAIN FAILED $TASK/dp$SUF" &
done
wait
echo "=== $TASK dp training done; static sweeps ==="
pids=()
for SEED in 0 1 2; do
    SUF=""; [ $SEED -gt 0 ] && SUF="_seed$SEED"
    $PY sweep_driver.py $TASK --methods=dp --tag="$SUF" --no-summary > logs/dp_eval_${TASK}$SUF.log 2>&1 & pids+=($!); sleep 5
done
for p in "${pids[@]}"; do wait $p || echo "SWEEP FAILED (pid $p)"; done
$PY static_summary.py $TASK
echo "=== $TASK DP_DONE ==="
