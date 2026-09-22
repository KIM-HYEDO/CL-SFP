"""Static (drift-free) results across training seeds, for any robomimic task.

final_summary.py and paired_test.py both assume a drift curve exists: they read
perturb*_s100_episodes.json and pool over a perturbation band. transport and
tool_hang are so far measured at perturb 0.0 only, so there is no curve to pool
and nothing for those scripts to read. This reads the checkpoint sweep instead
and reports the one number that exists: success at no disturbance, aggregated
over training seeds.

Checkpoint choice follows the project's protocol - argmax of the static score
over the sweep, per seed, which is also what sweep_driver reports. The chosen
epochs are printed because they are not comparable across methods when the
sweep lengths differ (see HANDOFF.md on the ah-axis budget problem; the same
trap applies here the moment early stopping lets one method see ep700-1000 and
another stop at 600).

Usage: python static_summary.py [task ...]      (default: transport tool_hang)
Writes outputs/static_summary.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = ("", "_seed1", "_seed2")
SEED_EVAL = 100

# display name -> checkpoint directory prefix
METHODS = {
    "SFP": "sfp",
    "CL-SFP": "cl_sfp",
    "CL-SFP+i": "cl_sfp_interp",
}

TASKS = tuple(sys.argv[1:]) or ("transport", "tool_hang")


def best_static(task, prefix, suffix):
    """(epoch, score) of the checkpoint with the highest drift-free score."""
    f = (ROOT / "outputs" / task / f"{prefix}{suffix}" / "eval"
         / f"sweep_s{SEED_EVAL}_episodes.json")
    if not f.is_file():
        return None
    payload = json.load(open(f))
    best = None
    for ck, entry in payload.get("checkpoints", {}).items():
        r = entry.get("results", {}).get("0.0")
        if r is None:
            continue
        # ties go to the earlier epoch, matching sweep_driver
        if best is None or (r["mean"], -int(ck)) > (best[1], -best[0]):
            best = (int(ck), r["mean"])
    return best


def main():
    rows = []
    for task in TASKS:
        for name, prefix in METHODS.items():
            picks = [best_static(task, prefix, s) for s in SEEDS]
            picks = [p for p in picks if p is not None]
            if not picks:
                continue
            eps = [p[0] for p in picks]
            sc = np.array([p[1] for p in picks])
            rows.append({
                "task": task, "method": name, "n_seeds": len(sc),
                "mean": float(sc.mean()),
                "std": float(sc.std(ddof=1)) if len(sc) > 1 else 0.0,
                "ckpts": "|".join(str(e) for e in eps),
            })

    if not rows:
        print("nothing measured yet for " + ", ".join(TASKS))
        return

    width = max(len(r["method"]) for r in rows)
    print(f"{'task':<11} {'method':<{width}} {'n':>2}  {'static':>16}  ckpts")
    print("-" * (40 + width))
    for r in rows:
        print(f"{r['task']:<11} {r['method']:<{width}} {r['n_seeds']:>2}  "
              f"{r['mean']:.3f} +/- {r['std']:.3f}  {r['ckpts']}")

    out = ROOT / "outputs" / "static_summary.csv"
    cols = ["task", "method", "n_seeds", "mean", "std", "ckpts"]
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float)
                              else str(r[c]) for c in cols) + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
