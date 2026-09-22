#!/usr/bin/env bash
# SFP-1step drift sweep for a seed, at that seed's best static checkpoint
# (taken from the existing sweep_s100_episodes.json). Usage: bash onestep_drift.sh <task> <tag>
set -e
TASK=$1; TAG=$2
cd /workspace/CL-SFP
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PY=.venv/bin/python
D=outputs/$TASK/sfp_1step$TAG/eval
BEST=$($PY - <<PY
import json
d = json.load(open("$D/sweep_s100_episodes.json"))["checkpoints"]
print(max(d, key=lambda k: (d[k]["results"]["0.0"]["mean"], -int(k))))
PY
)
echo "==== $TASK sfp_1step$TAG best static ckpt = ep$BEST ===="
$PY algo/sfp_1step.py --task $TASK --mode eval --ckpt $BEST --seeds 100 --tag "$TAG" \
    --perturbs 0.0,0.0001,0.00025,0.0003,0.0005,0.00075,0.001,0.00125,0.0015 \
    --out-json $D/perturb_s100.json
echo "=== done $TASK sfp_1step$TAG ==="
