#!/usr/bin/env bash
# SFP hyper-parameter grid on tool_hang, one seed: the flow constants and the
# number of gradient steps, everything else the shared recipe. Each config
# trains under its own tag, then all static sweeps run side by side.
#
#   _s0.2      sigma0 0.2            less initial noise (paper 0.4)
#   _s0.1      sigma0 0.1
#   _k14       k 14                  Euler-deadbeat contraction at dt=1/14 (paper 10)
#   _k5        k 5                   slower contraction
#   _b256      batch 256             4x the gradient steps of batch 1024
#   _s0.2_k14  both flow changes
#
# Usage: bash hp_sweep_toolhang.sh        (uses both GPUs)
set -e
cd /workspace/CL-SFP; mkdir -p logs
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
train() {  # gpu tag flags
    [ -f outputs/tool_hang/sfp$2/ep1000.ckpt ] && { echo "skip train sfp$2"; return; }
    CUDA_VISIBLE_DEVICES=$1 $PY algo/sfp.py --task tool_hang --mode train --epochs 1000 --train-seed 0 --tag "$2" $3 \
        > logs/hp_train_tool_hang$2.log 2>&1 && echo "trained sfp$2" || echo "TRAIN FAILED sfp$2"
}
train 0 _s0.2     "--sigma0 0.2" &
train 0 _s0.1     "--sigma0 0.1" &
train 0 _k14      "--k 14" &
train 1 _k5       "--k 5" &
train 1 _b256     "--batch-size 256" &
train 1 _s0.2_k14 "--sigma0 0.2 --k 14" &
wait
echo "=== training done; sweeps ==="
pids=(); i=0
for TAG in _s0.2 _s0.1 _k14 _k5 _b256 _s0.2_k14; do
    CUDA_VISIBLE_DEVICES=$((i % 2)) $PY sweep_driver.py tool_hang --methods=sfp --tag="$TAG" --no-summary \
        > logs/hp_eval_tool_hang$TAG.log 2>&1 & pids+=($!); i=$((i+1)); sleep 5
done
for p in "${pids[@]}"; do wait $p || echo "SWEEP FAILED (pid $p)"; done
echo "=== HP_DONE ==="
