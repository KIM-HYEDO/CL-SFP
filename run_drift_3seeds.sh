#!/usr/bin/env bash
# Drift curves for transport and tool_hang: 3 training seeds x 3 methods, at
# each model's statically selected checkpoint, 100 eval seeds.
#
# Levels come from the 2026-09-23 one-seed pilot (perturb_pilot_*_s50.json):
#   transport  moves between 1e-4 and 5e-4 and is at floor by 1e-3
#   tool_hang  is flat to 2.5e-4, falls through 1e-3 (frame drift)
# Both bands sit BELOW the can/square band (1e-4..1.5e-3): the same per-step
# drift accumulates 2.8x further over a 700-step episode.
#
# tool_hang drifts the FRAME (tasks.PERTURB_OBJECT default). The pilot also
# tried the stand: every method is at 0.00 from 5e-5 on, because 0.02 mm/step
# over 700 steps moves the insertion target 1.4 cm, so it measures how far the
# target moved from the demonstrations, not how the policy reacts.
#
# Usage: bash run_drift_3seeds.sh            (both tasks, one GPU each)
set -e
cd /workspace/CL-SFP; mkdir -p logs
LV_TRANSPORT="0.0,0.00005,0.0001,0.00025,0.0005,0.00075,0.001"
LV_TOOLHANG="0.0,0.0001,0.00025,0.0005,0.00075,0.001,0.0015"
N=100
run() {  # gpu task dir levels
    CUDA_VISIBLE_DEVICES=$1 bash drift_eval.sh $2 $3 $4 $N > logs/drift_$2_$3.log 2>&1 &
    echo "launched $2 $3 (pid $!)"; sleep 5
}
for SUF in "" _seed1 _seed2; do
    for M in sfp cl_sfp cl_sfp_interp; do
        run 0 transport $M$SUF $LV_TRANSPORT
        run 1 tool_hang $M$SUF $LV_TOOLHANG
    done
done
wait; echo DRIFT_DONE
