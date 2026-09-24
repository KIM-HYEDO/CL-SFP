#!/usr/bin/env bash
# Drift curve for one (task, method-dir) at its statically selected checkpoint.
#
# The checkpoint is the static argmax already chosen by the sweep in
# outputs/<task>/<dir>/eval/sweep_s100_episodes.json - the same protocol as
# fair_eval.sh, so the drift curve and the static table describe one model.
#
# TARGET (robomimic multi-object tasks only) overrides tasks.PERTURB_OBJECT for
# this run and goes into the output filename, because drifting a different
# object is a different experiment; the sweep file records it too.
#
# Usage: [ABS=1] bash drift_eval.sh <task> <dir> <levels> <n_eval_seeds> [target] [name]
#   ABS=1 for absolute-action checkpoints (dirs tagged _abs...)
#   e.g. bash drift_eval.sh tool_hang cl_sfp_interp 0.0,0.0001,0.0005 50 stand pilot
#   -> outputs/tool_hang/cl_sfp_interp/eval/perturb_pilot_stand_s50.json
set -e
cd /workspace/CL-SFP
TASK=$1; DIR=$2; LEVELS=$3; N=$4; TARGET=${5:-}; NAME=${6:-}
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python

# dir -> (script, tag): sfp_seed1 -> sfp/_seed1 ; cl_sfp_interp_seed2 -> cl_sfp/_interp_seed2
case "$DIR" in
    sfp*)    METHOD=sfp;    TAG=${DIR#sfp} ;;
    cl_sfp*) METHOD=cl_sfp; TAG=${DIR#cl_sfp} ;;
    *) echo "unknown dir $DIR"; exit 1 ;;
esac
D=outputs/$TASK/$DIR/eval
BEST=$($PY - <<PY
import json
d = json.load(open("$D/sweep_s100_episodes.json"))["checkpoints"]
print(max((k for k in d if d[k]["results"].get("0.0")),
          key=lambda k: (d[k]["results"]["0.0"]["mean"], -int(k))))
PY
)
OUT=$D/perturb${NAME:+_$NAME}${TARGET:+_$TARGET}_s$N.json
echo "==== $TASK/$DIR  ckpt=ep$BEST  target=${TARGET:-default}  -> $OUT"
CLSFP_PERTURB_OBJECT=$TARGET $PY algo/$METHOD.py --task $TASK --mode eval --ckpt $BEST \
    --seeds $N --perturbs $LEVELS --tag "$TAG" --out-json $OUT ${ABS:+--abs-action}
