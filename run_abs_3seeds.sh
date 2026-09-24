#!/usr/bin/env bash
# Absolute-action recipe test: does Diffusion Policy's action space close the
# gap to its numbers? square (DP-C 0.96 avg) and tool_hang (0.93), three
# methods x three seeds, trained on actions_abs (pos + 6D rotation + gripper)
# with the OSC controller in control_delta=False, then the static sweep.
#
# Everything else is held fixed against the delta-action runs: same UNet, same
# epochs, same horizons, same 250/700-step budgets, same checkpoint rule.
# Checkpoints land in outputs/<task>/{sfp,cl_sfp}_abs[_interp][_seedN]/.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_abs_3seeds.sh square
set -e
cd /workspace/CL-SFP; mkdir -p logs
TASK=$1; [ -n "$TASK" ] || { echo "usage: $0 <task>"; exit 1; }
export MUJOCO_GL=egl LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
# three trainings at a time per GPU: one saturates ~80%, three share it well
train() {  # script tag flags
    [ -f outputs/$TASK/$1$2/ep1000.ckpt ] && { echo "skip train $TASK/$1$2"; return; }
    $PY algo/$1.py --task $TASK --mode train --epochs 1000 --abs-action --tag "$2" $3 \
        > logs/abs_train_${TASK}_$1$2.log 2>&1 && echo "trained $TASK/$1$2" || echo "TRAIN FAILED $TASK/$1$2"
}
for SEED in 0 1 2; do
    SUF=""; [ $SEED -gt 0 ] && SUF="_seed$SEED"
    train sfp    "_abs$SUF"        "--train-seed $SEED" &
    train cl_sfp "_abs$SUF"        "--train-seed $SEED" &
    train cl_sfp "_abs_interp$SUF" "--train-seed $SEED --cond-interp" &
    wait
done
echo "=== $TASK abs training done; static sweeps ==="
pids=()
for SEED in 0 1 2; do
    SUF=""; [ $SEED -gt 0 ] && SUF="_seed$SEED"
    for SPEC in "sfp|_abs$SUF" "cl_sfp|_abs$SUF" "cl_sfp|_abs_interp$SUF"; do
        IFS='|' read -r M TAG <<< "$SPEC"
        $PY sweep_driver.py $TASK --methods=$M --tag="$TAG" --no-summary --abs-action \
            > logs/abs_eval_${TASK}_$M$TAG.log 2>&1 & pids+=($!); sleep 5
    done
done
for p in "${pids[@]}"; do wait $p || echo "SWEEP FAILED (pid $p)"; done
$PY static_summary.py $TASK --variant=abs
echo "=== $TASK ABS_DONE ==="
