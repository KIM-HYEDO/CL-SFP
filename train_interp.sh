#!/usr/bin/env bash
# CL-SFP with interpolated (continuously time-aligned) conditioning windows.
# Train 3 seeds on the given task, then evaluate.
# Usage: bash train_interp.sh pusht      (random-direction drift curve, ep1000)
#        bash train_interp.sh can        (checkpoint sweep at drift 0)
set -e
TASK=$1
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export PUSHT_PERTURB_DIR=random
PYTHON=".venv/bin/python"
LEVELS="0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8"

for SEED in 0 1 2; do
    TAG="_interp"; [ $SEED -gt 0 ] && TAG="_interp_seed$SEED"
    if [ -f outputs/$TASK/cl_sfp$TAG/ep1000.ckpt ]; then echo "skip train $TASK$TAG"; continue; fi
    echo "==== train cl_sfp(interp) $TASK seed=$SEED ===="
    $PYTHON algo/cl_sfp.py --task $TASK --mode train --epochs 1000 --cond-interp \
        --train-seed $SEED --tag $TAG
done

for SEED in 0 1 2; do
    TAG="_interp"; [ $SEED -gt 0 ] && TAG="_interp_seed$SEED"
    echo "==== eval cl_sfp(interp) $TASK seed=$SEED ===="
    if [ "$TASK" = pusht ]; then
        $PYTHON algo/cl_sfp.py --task pusht --mode eval --ckpt 1000 --seeds 100 \
            --perturbs $LEVELS --tag $TAG \
            --out-json outputs/pusht/cl_sfp$TAG/eval/perturb_rand_s100.json
    else
        $PYTHON sweep_driver.py $TASK --methods=cl_sfp --tag=$TAG
    fi
done
echo "=== interp $TASK complete ==="
