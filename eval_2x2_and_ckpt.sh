#!/usr/bin/env bash
# (a) 2x2 completion: CL-SFP weights + replan-every-step, all seeds/tasks.
# (b) Push-T checkpoint selection at drift 0 and 1.4 px, SFP & CL-SFP, 3 seeds.
set -e
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export PUSHT_PERTURB_DIR=random
PYTHON=".venv/bin/python"
LEVELS="0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8"

case $1 in
  cl1step)
    for TAG in "" _seed1 _seed2; do
        echo "==== cl_sfp_1step pusht tag='$TAG' ===="
        $PYTHON algo/cl_sfp_1step.py --task pusht --mode eval --ckpt 1000 --seeds 100 \
            --perturbs $LEVELS --tag "$TAG" \
            --out-json outputs/pusht/cl_sfp_1step$TAG/eval/perturb_rand_s100.json
    done
    for TAG in "" _seed1 _seed2; do
        echo "==== cl_sfp_1step robomimic tag='$TAG' ===="
        $PYTHON sweep_driver.py can lift square --methods=cl_sfp_1step --tag=$TAG
    done
    ;;
  ckpt)
    for TAG in "" _seed1 _seed2; do
        for METHOD in sfp cl_sfp; do
            echo "==== pusht ckpt sweep $METHOD tag='$TAG' ===="
            $PYTHON algo/$METHOD.py --task pusht --mode eval --ckpt 200,400,600,800,1000 \
                --seeds 100 --perturbs 0.0,1.4 --tag "$TAG" \
                --out-json outputs/pusht/$METHOD$TAG/eval/ckpt_rand_s100.json
        done
    done
    ;;
  *) echo "usage: $0 {cl1step|ckpt}"; exit 1 ;;
esac
echo "=== $1 complete ==="
