#!/usr/bin/env bash
# Full pipeline: train sfp + cl_sfp, then eval both on pusht
set -e
cd /workspace/CL-SFP
export MUJOCO_GL=egl

VENV=".venv/bin/python"

echo "========================================"
echo "[1/4] Training SFP (pusht, 1000 epochs)"
echo "========================================"
$VENV algo/sfp.py --task pusht --mode train --epochs 1000

echo "=========================================="
echo "[2/4] Training CL-SFP (pusht, 1000 epochs)"
echo "=========================================="
$VENV algo/cl_sfp.py --task pusht --mode train --epochs 1000

echo "============================"
echo "[3/4] Eval SFP (pusht)"
echo "============================"
$VENV algo/sfp.py --task pusht --mode eval --ckpt 1000 --seeds 100 --perturbs 0.0-2.0

echo "=============================="
echo "[4/4] Eval CL-SFP (pusht)"
echo "=============================="
$VENV algo/cl_sfp.py --task pusht --mode eval --ckpt 1000 --seeds 100 --perturbs 0.0-2.0

echo ""
echo "=== Pipeline complete ==="
