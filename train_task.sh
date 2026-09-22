#!/usr/bin/env bash
# Usage: bash train_task.sh <task>
set -e
TASK=$1
cd /workspace/CL-SFP
PYTHON=".venv/bin/python"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH

echo "========================================"
echo " Training SFP  -- task: $TASK"
echo "========================================"
$PYTHON algo/sfp.py --task $TASK --mode train --epochs 1000

echo "========================================"
echo " Training CL-SFP -- task: $TASK"
echo "========================================"
$PYTHON algo/cl_sfp.py --task $TASK --mode train --epochs 1000

echo "=== $TASK training complete ==="
