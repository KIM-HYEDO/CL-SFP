#!/usr/bin/env bash
# Train SFP and CL-SFP with additional seeds on every task, then evaluate:
#   robomimic -> checkpoint sweep (ep100-600 + early stop) at perturb 0
#   pusht     -> ep1000 perturbation curve
# Usage: bash train_seeds.sh 1 2
set -e
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PYTHON=".venv/bin/python"

for SEED in "$@"; do
    TAG="_seed$SEED"
    for TASK in pusht can lift square; do
        for METHOD in sfp cl_sfp; do
            if [ -f outputs/$TASK/$METHOD$TAG/ep1000.ckpt ]; then
                echo "skip train $TASK/$METHOD$TAG (exists)"; continue
            fi
            echo "==== train $TASK / $METHOD  seed=$SEED ===="
            $PYTHON algo/$METHOD.py --task $TASK --mode train --epochs 1000 \
                --train-seed $SEED --tag $TAG
        done
    done
done

for SEED in "$@"; do
    TAG="_seed$SEED"
    echo "==== eval pusht seed=$SEED  (random drift direction) ===="
    for METHOD in sfp cl_sfp sfp_1step; do
        PUSHT_PERTURB_DIR=random $PYTHON algo/$METHOD.py --task pusht --mode eval \
            --ckpt 1000 --seeds 100 --tag $TAG \
            --perturbs 0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8 \
            --out-json outputs/pusht/$METHOD$TAG/eval/perturb_rand_s100.json
    done
    echo "==== eval robomimic seed=$SEED ===="
    $PYTHON sweep_driver.py can lift square --methods=sfp,cl_sfp,sfp_1step --tag=$TAG
done
echo "=== seeds $* complete ==="
