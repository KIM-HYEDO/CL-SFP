#!/usr/bin/env bash
# Train and evaluate the long-horizon tasks: transport (two-arm) and tool_hang.
#
# Scope, as asked for:
#   methods   sfp, cl_sfp, cl_sfp+interp   (no 1step / gated / ah variants)
#   drift     none - everything is scored at perturb 0.0
#
# So the evaluation here is exactly sweep_driver's static checkpoint sweep:
# ep100-600, then one checkpoint at a time past 600 only while the last one set
# a new best. No drift curve is produced; adding one later is a separate pass
# over the chosen checkpoints and does not invalidate anything written here.
#
# Episodes run to the task's own 700-step budget rather than the 250 the three
# original tasks were measured at (env/robomimic/tasks.py MAX_STEPS), so these
# numbers are NOT comparable with can/lift/square on absolute success rate.
# robomimic sets ignore_done, so every episode costs the full budget.
#
# Usage: bash train_newtasks.sh [task ...]        (default: transport tool_hang)
#        SEEDS="0" bash train_newtasks.sh transport      (pilot on one seed)
set -e
cd /workspace/CL-SFP
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/workspace/IKEA-Assembly/.native-cuda/lib:$LD_LIBRARY_PATH
PYTHON=".venv/bin/python"
TASKS=${@:-"transport tool_hang"}
SEEDS=${SEEDS:-"0 1 2"}

for TASK in $TASKS; do
    for SEED in $SEEDS; do
        SUF=""; [ "$SEED" -gt 0 ] && SUF="_seed$SEED"
        # (script, checkpoint dir tag, extra train flags)
        for SPEC in "sfp|$SUF|" "cl_sfp|$SUF|" "cl_sfp|_interp$SUF|--cond-interp"; do
            IFS='|' read -r METHOD TAG FLAGS <<< "$SPEC"
            if [ -f outputs/$TASK/$METHOD$TAG/ep1000.ckpt ]; then
                echo "skip train $TASK/$METHOD$TAG (exists)"; continue
            fi
            echo "==== train $TASK / $METHOD$TAG  seed=$SEED ===="
            $PYTHON algo/$METHOD.py --task $TASK --mode train --epochs 1000 \
                --train-seed $SEED --tag "$TAG" $FLAGS
        done
    done

    for SEED in $SEEDS; do
        SUF=""; [ "$SEED" -gt 0 ] && SUF="_seed$SEED"
        echo "==== sweep $TASK sfp,cl_sfp  seed=$SEED ===="
        $PYTHON sweep_driver.py $TASK --methods=sfp,cl_sfp --tag="$SUF"
        echo "==== sweep $TASK cl_sfp+interp  seed=$SEED ===="
        $PYTHON sweep_driver.py $TASK --methods=cl_sfp --tag="_interp$SUF"
    done
done
echo "=== $TASKS complete ==="
