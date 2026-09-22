#!/usr/bin/env bash
# Train a CL-SFP variant on one task for seeds 0-2, then fair-evaluate each.
# Usage: bash train_variant.sh <task> <tag> <extra train args...>
#   e.g. bash train_variant.sh can _interp_smin0.02 --cond-interp --sigma-min 0.02
set -e
TASK=$1; TAG=$2; shift 2
cd /workspace/CL-SFP
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PY=.venv/bin/python
for SEED in 0 1 2; do
    T="$TAG"; [ $SEED -gt 0 ] && T="${TAG}_seed$SEED"
    if [ -f outputs/$TASK/cl_sfp$T/ep1000.ckpt ]; then echo "skip train $TASK$T"; continue; fi
    echo "==== train cl_sfp $TASK tag=$T  ($*) ===="
    $PY algo/cl_sfp.py --task $TASK --mode train --epochs 1000 --train-seed $SEED --tag $T "$@"
done
for SEED in 0 1 2; do
    T="$TAG"; [ $SEED -gt 0 ] && T="${TAG}_seed$SEED"
    bash fair_eval.sh $TASK cl_sfp $T 8
done
echo "=== variant $TAG on $TASK complete ==="
