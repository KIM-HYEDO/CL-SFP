#!/usr/bin/env bash
# Evaluate one (task, method, tag, action-horizon) with fair checkpoint choice:
#   robomimic: static sweep ep100-600 at this horizon -> best -> drift sweep
#   pusht:     ep1000, random-direction drift curve
# Usage: bash fair_eval.sh <task> <sfp|cl_sfp> <tag> <ah>
set -e
TASK=$1; METHOD=$2; TAG=$3; AH=${4:-8}
cd /workspace/CL-SFP
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export PUSHT_PERTURB_DIR=random
PY=.venv/bin/python
D=outputs/$TASK/$METHOD$TAG/eval
AHS=""; [ "$AH" != 8 ] && AHS="_ah$AH"

if [ "$TASK" = pusht ]; then
    $PY algo/$METHOD.py --task pusht --mode eval --ckpt 1000 --seeds 100 --tag "$TAG" \
        --action-horizon $AH \
        --perturbs 0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8 \
        --out-json $D/perturb_rand${AHS}_s100.json
else
    $PY algo/$METHOD.py --task $TASK --mode eval --ckpt 100-600 --seeds 100 --perturbs 0.0 \
        --action-horizon $AH --tag "$TAG"
    BEST=$($PY - <<PY
import json
d = json.load(open("$D/sweep_s100${AHS}_episodes.json"))["checkpoints"]
print(max(d, key=lambda k: (d[k]["results"]["0.0"]["mean"], -int(k))))
PY
)
    echo "==== $TASK $METHOD$TAG ah$AH: best static ckpt = ep$BEST ===="
    $PY algo/$METHOD.py --task $TASK --mode eval --ckpt $BEST --seeds 100 --tag "$TAG" \
        --action-horizon $AH \
        --perturbs 0.0,0.0001,0.00025,0.0003,0.0005,0.00075,0.001,0.00125,0.0015 \
        --out-json $D/perturb${AHS}_s100.json
fi
echo "=== done $TASK $METHOD$TAG ah$AH ==="
