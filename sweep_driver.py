"""Checkpoint sweep for robomimic tasks with early stopping.

Evaluate ep100..ep600 for every (task, method). Past ep600, continue one
checkpoint at a time only while the most recent checkpoint set a new best;
stop as soon as it fails to. Already-measured checkpoints are reused by the
eval scripts' resume logic.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv/bin/python"
TASKS = tuple(a for a in sys.argv[1:] if not a.startswith("--")) or ("can", "lift", "square")
METHODS = tuple(next((a.split("=", 1)[1] for a in sys.argv[1:]
                      if a.startswith("--methods=")), "sfp,cl_sfp").split(","))
TAG = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "")
SEEDS = 100
PERTURB = "0.0"
MANDATORY_END = 600
LAST_EP = 1000

ENV = dict(os.environ,
           MUJOCO_GL="egl",
           LD_LIBRARY_PATH="/workspace/IKEA-Assembly/.native-cuda/lib:"
                           + os.environ.get("LD_LIBRARY_PATH", ""))


def run_eval(task, method, ckpt_spec):
    cmd = [str(PY), str(ROOT / f"algo/{method}.py"), "--task", task,
           "--mode", "eval", "--ckpt", ckpt_spec,
           "--seeds", str(SEEDS), "--perturbs", PERTURB, "--tag", TAG]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def scores(task, method):
    f = ROOT / "outputs" / task / f"{method}{TAG}" / "eval" / f"sweep_s{SEEDS}_episodes.json"
    if not f.is_file():
        return {}
    payload = json.load(open(f))
    out = {}
    for ck, e in payload.get("checkpoints", {}).items():
        r = e.get("results", {}).get(PERTURB)
        if r:
            out[int(ck)] = (r["mean"], r["ci95"])
    return out


def sweep(task, method):
    run_eval(task, method, f"100-{MANDATORY_END}")
    ep = MANDATORY_END
    while ep < LAST_EP:
        s = scores(task, method)
        best_before = max(v[0] for k, v in s.items() if k < ep)
        if s[ep][0] < best_before:
            print(f"[{task}/{method}] ep{ep}={s[ep][0]:.3f} < best {best_before:.3f}"
                  f" -> skipping ep{ep + 100}..ep{LAST_EP}", flush=True)
            break
        ep += 100
        run_eval(task, method, str(ep))


def main():
    if "--summary-only" not in sys.argv:
        for task in TASKS:
            for method in METHODS:
                sweep(task, method)

    rows = []
    # The methods actually asked for, plus the three the summary has always
    # covered, so a run with --methods=cl_sfp still reports what is on disk and
    # an old invocation keeps producing exactly the rows it used to. scores()
    # returns {} for anything unmeasured, so a superset costs nothing.
    summary_methods = list(dict.fromkeys(
        list(METHODS) + ["sfp", "cl_sfp", "sfp_1step"]))
    for task in TASKS:
        for method in summary_methods:
            s = scores(task, method)
            if not s:
                continue
            best_ep = max(s, key=lambda k: (s[k][0], -k))
            rows.append({"task": task, "method": method,
                         "best_ckpt": best_ep,
                         "score": s[best_ep][0], "ci95": s[best_ep][1],
                         "evaluated": sorted(s)})

    out_json = ROOT / "outputs" / f"robomimic_best_ckpt{TAG}.json"
    out_csv = ROOT / "outputs" / f"robomimic_best_ckpt{TAG}.csv"
    json.dump(rows, open(out_json, "w"), indent=2)
    with open(out_csv, "w") as fh:
        fh.write("task,method,best_ckpt,score,ci95\n")
        for r in rows:
            fh.write(f"{r['task']},{r['method']},{r['best_ckpt']},"
                     f"{r['score']:.4f},{r['ci95']:.4f}\n")

    print("\n  task     method   best_ckpt   score    95% CI   evaluated")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['task']:<8} {r['method']:<8} {r['best_ckpt']:>9}   "
              f"{r['score']:.4f}   {r['ci95']:.4f}   {r['evaluated']}")
    print(f"\nsaved -> {out_json}\nsaved -> {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
