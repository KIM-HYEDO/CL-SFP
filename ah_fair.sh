#!/usr/bin/env bash
# Fair replan-horizon baseline: pick the checkpoint that is best for THIS
# horizon at drift 0 (ep100-1000), then run the drift sweep at that checkpoint.
# Usage: bash ah_fair.sh <task> <ah> [tag]
set -e
TASK=$1; AH=$2; TAG=${3:-}
cd /workspace/CL-SFP
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PY=.venv/bin/python
LV="0.0,0.0001,0.00025,0.0003,0.0005,0.00075,0.001,0.00125,0.0015"
D=outputs/$TASK/sfp$TAG/eval

$PY algo/sfp.py --task $TASK --mode eval --ckpt 100-1000 --seeds 100 --perturbs 0.0 \
    --action-horizon $AH --tag "$TAG"
BEST=$($PY - <<PY
import json
d = json.load(open("$D/sweep_s100_ah${AH}_episodes.json"))["checkpoints"]
print(max(d, key=lambda k: (d[k]["results"]["0.0"]["mean"], -int(k))))
PY
)
echo "==== $TASK sfp$TAG ah$AH: best static ckpt = ep$BEST ===="
for f in perturb_ah${AH}_s100_episodes.json perturb_ah${AH}_s100.json perturb_ah${AH}_s100.csv; do
    [ -f $D/$f ] && mv $D/$f $D/${f%.*}.sfpckpt.${f##*.}
done
$PY algo/sfp.py --task $TASK --mode eval --ckpt $BEST --seeds 100 --perturbs $LV \
    --action-horizon $AH --tag "$TAG" --out-json $D/perturb_ah${AH}_s100.json
echo "=== done $TASK ah$AH tag='$TAG' ==="
