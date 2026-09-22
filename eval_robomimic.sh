#!/usr/bin/env bash
# Eval SFP + CL-SFP on can, lift, square
set -e
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PYTHON=".venv/bin/python"

for TASK in can lift square; do
    echo "========================================"
    echo " Eval SFP  -- task: $TASK"
    echo "========================================"
    $PYTHON algo/sfp.py --task $TASK --mode eval --ckpt 1000 --seeds 100 --perturbs 0.0

    echo "========================================"
    echo " Eval CL-SFP -- task: $TASK"
    echo "========================================"
    $PYTHON algo/cl_sfp.py --task $TASK --mode eval --ckpt 1000 --seeds 100 --perturbs 0.0
done

echo "=== All eval complete ==="
