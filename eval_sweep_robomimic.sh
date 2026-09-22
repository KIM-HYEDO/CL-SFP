#!/usr/bin/env bash
# Sweep all checkpoints (ep100~ep1000) for can, lift, square -- SFP and CL-SFP
set -e
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PYTHON=".venv/bin/python"

for TASK in can lift square; do
    echo "========================================"
    echo " Sweep SFP  -- task: $TASK"
    echo "========================================"
    $PYTHON algo/sfp.py --task $TASK --mode eval --ckpt auto --seeds 100 --perturbs 0.0

    echo "========================================"
    echo " Sweep CL-SFP -- task: $TASK"
    echo "========================================"
    $PYTHON algo/cl_sfp.py --task $TASK --mode eval --ckpt auto --seeds 100 --perturbs 0.0
done

echo "=== All sweeps complete ==="
