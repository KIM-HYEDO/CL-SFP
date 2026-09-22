#!/usr/bin/env bash
# Object-drift perturbation sweep on robomimic, each method at its best
# perturb-0 checkpoint. Usage: bash eval_perturb_robomimic.sh <can|square>
set -e
TASK=$1
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PYTHON=".venv/bin/python"
LEVELS="${LEVELS:-0.0,0.0005,0.001,0.0015,0.002,0.003,0.004}"   # m/step; override via env

case $TASK in
    can)    declare -A CKPT=([sfp]=400 [sfp_1step]=500 [cl_sfp]=200) ;;
    square) declare -A CKPT=([sfp]=300 [sfp_1step]=800 [cl_sfp]=500) ;;
    *) echo "unknown task $TASK"; exit 1 ;;
esac

for METHOD in sfp sfp_1step cl_sfp; do
    echo "========================================"
    echo " Perturb sweep  $TASK / $METHOD  ep${CKPT[$METHOD]}"
    echo "========================================"
    $PYTHON algo/$METHOD.py --task $TASK --mode eval --ckpt ${CKPT[$METHOD]} \
        --seeds 100 --perturbs $LEVELS \
        --out-json outputs/$TASK/$METHOD/eval/perturb_s100.json
done
echo "=== $TASK perturb sweep complete ==="
