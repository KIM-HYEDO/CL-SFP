#!/usr/bin/env bash
# Static checkpoint sweeps for one task, all nine (method, seed) combinations
# at once.
#
# Evaluation is MuJoCo-bound and single-threaded: one sweep pins one CPU core
# and leaves the GPU at ~5%. Running the nine combinations in sequence, as
# train_newtasks.sh first did, would take ~40 h on transport while 126 of the
# machine's 128 cores sat idle. Side by side they finish in the time of the
# longest one (~5-7 h), and the sweeps are independent - each writes its own
# outputs/<task>/<method><tag>/eval/ - so nothing needs coordinating beyond
# the summary, which runs once at the end.
#
# Resumable: the eval scripts skip checkpoints already in their sweep file.
#
# Thread limits: this container caps the process tree at 4096 pids, and with
# 128 cores every OpenMP pool (numpy, torch, MuJoCo) defaults to 128 threads,
# so an unconstrained eval process carries 130-380 threads and nine of them
# starting at once blow through the cap - half die inside MjModel compilation
# with "Caught an unknown exception!" or "libgomp: Thread creation failed".
# The work is a single-threaded simulator plus batch-1 inference, so one
# thread per pool loses nothing. Starts are also staggered, and a sweep that
# still fails is retried once after the rest are up.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash eval_newtasks_parallel.sh transport
set -e
cd /workspace/CL-SFP
TASK=$1
[ -n "$TASK" ] || { echo "usage: $0 <task>"; exit 1; }
PYTHON=".venv/bin/python"
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STAGGER=${STAGGER:-5}

pids=(); names=(); methods=(); tags=()
for SEED in 0 1 2; do
    SUF=""; [ "$SEED" -gt 0 ] && SUF="_seed$SEED"
    for SPEC in "sfp|$SUF" "cl_sfp|$SUF" "cl_sfp|_interp$SUF"; do
        IFS='|' read -r METHOD TAG <<< "$SPEC"
        LOG=logs/eval_${TASK}_${METHOD}${TAG}.log
        echo "launch $TASK / $METHOD$TAG  -> $LOG"
        $PYTHON sweep_driver.py $TASK --methods=$METHOD --tag="$TAG" --no-summary \
            > "$LOG" 2>&1 &
        pids+=($!); names+=("$METHOD$TAG"); methods+=("$METHOD"); tags+=("$TAG")
        sleep "$STAGGER"
    done
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "done   $TASK / ${names[$i]}"; continue; fi
    echo "retry  $TASK / ${names[$i]}  (first attempt failed, see logs/eval_${TASK}_${names[$i]}.log)"
    mv "logs/eval_${TASK}_${names[$i]}.log" "logs/eval_${TASK}_${names[$i]}.attempt1.log"
    if $PYTHON sweep_driver.py $TASK --methods="${methods[$i]}" --tag="${tags[$i]}" --no-summary \
            > "logs/eval_${TASK}_${names[$i]}.log" 2>&1; then
        echo "done   $TASK / ${names[$i]}  (on retry)"
    else
        echo "FAILED $TASK / ${names[$i]}  (see logs/eval_${TASK}_${names[$i]}.log)"; failed=1
    fi
done

$PYTHON static_summary.py $TASK
if [ $failed = 0 ]; then echo "=== $TASK eval complete ==="
else echo "=== $TASK eval finished with failures ==="; exit 1; fi
