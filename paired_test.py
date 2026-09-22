"""Paired comparison of methods from the per-episode eval records.

Every method is scored on the same episode seeds, so differences are paired.
Robomimic success is binary -> exact McNemar (binomial test on discordant
pairs). Push-T coverage is continuous -> Wilcoxon signed-rank plus a paired
bootstrap CI on the mean difference. Both are reported per perturbation level
and pooled over the "mid-disturbance" band where the curves separate.

Usage: python paired_test.py            (writes outputs/paired_tests.csv)
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
SEEDS = 100
METHODS = ("sfp", "sfp_1step", "cl_sfp")
BEST = {"can": {"sfp": 400, "sfp_1step": 500, "cl_sfp": 200},
        "square": {"sfp": 300, "sfp_1step": 800, "cl_sfp": 500}}
# Perturbation band pooled for the headline test (units: px/step or m/step).
MID_BAND = {"pusht": (0.4, 1.0), "pusht_rand": (1.2, 2.0),
            "can": (0.0001, 0.00075), "square": (0.0003, 0.00125)}
rng = np.random.default_rng(0)


def load(task, method):
    if task == "pusht":
        f, ck = ROOT / f"outputs/pusht/{method}/eval/sweep_s{SEEDS}_episodes.json", "1000"
    elif task == "pusht_rand":
        f, ck = ROOT / f"outputs/pusht/{method}/eval/perturb_rand_s{SEEDS}_episodes.json", "1000"
    else:
        f, ck = ROOT / f"outputs/{task}/{method}/eval/perturb_s{SEEDS}_episodes.json", str(BEST[task][method])
    res = json.load(open(f))["checkpoints"][ck]["results"]
    return {float(p): np.asarray(v["scores"], dtype=float) for p, v in res.items()}


def mcnemar(a, b):
    """Exact McNemar on binary paired outcomes; returns (n_b_wins, n_a_wins, p)."""
    b_only = int(np.sum((b > 0.5) & (a <= 0.5)))
    a_only = int(np.sum((a > 0.5) & (b <= 0.5)))
    n = a_only + b_only
    p = 1.0 if n == 0 else stats.binomtest(b_only, n, 0.5).pvalue
    return b_only, a_only, p


def boot_ci(d, n=10000):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def compare(task, a_name, b_name, level, a, b):
    d = b - a
    row = {"task": task, "level": level, "a": a_name, "b": b_name,
           "mean_a": a.mean(), "mean_b": b.mean(), "diff": d.mean(), "n": len(d)}
    lo, hi = boot_ci(d)
    row["ci_lo"], row["ci_hi"] = lo, hi
    if task.startswith("pusht"):
        row["test"] = "wilcoxon"
        row["p"] = 1.0 if np.all(d == 0) else stats.wilcoxon(d, zero_method="zsplit").pvalue
    else:
        row["test"] = "mcnemar"
        bw, aw, p = mcnemar(a, b)
        row["p"] = p
        row["b_wins"], row["a_wins"] = bw, aw
    return row


def main():
    rows = []
    for task in ("pusht", "pusht_rand", "can", "square"):
        data = {m: load(task, m) for m in METHODS}
        levels = sorted(set.intersection(*(set(d) for d in data.values())))
        for a_name, b_name in combinations(METHODS, 2):
            for lv in levels:
                rows.append(compare(task, a_name, b_name, lv, data[a_name][lv], data[b_name][lv]))
            lo, hi = MID_BAND[task]
            band = [lv for lv in levels if lo - 1e-9 <= lv <= hi + 1e-9]
            a = np.concatenate([data[a_name][lv] for lv in band])
            b = np.concatenate([data[b_name][lv] for lv in band])
            r = compare(task, a_name, b_name, f"pooled[{band[0]:g}..{band[-1]:g}]", a, b)
            rows.append(r)

    out = ROOT / "outputs" / "paired_tests.csv"
    cols = ["task", "level", "a", "b", "n", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi",
            "test", "p", "a_wins", "b_wins"]
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                              for c in cols) + "\n")

    def star(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    print("\n=== Pooled over mid-disturbance band (b - a) ===")
    print(f"{'task':7}{'a':11}{'b':11}{'n':>5}{'mean a':>8}{'mean b':>8}{'diff':>8}"
          f"{'95% CI':>18}{'p':>9}")
    for r in rows:
        if str(r["level"]).startswith("pooled"):
            print(f"{r['task']:7}{r['a']:11}{r['b']:11}{r['n']:>5}{r['mean_a']:>8.3f}"
                  f"{r['mean_b']:>8.3f}{r['diff']:>+8.3f}"
                  f"   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{r['p']:>8.4f} {star(r['p'])}")

    print("\n=== Per level, CL-SFP vs SFP (b - a) ===")
    for task in ("pusht", "pusht_rand", "can", "square"):
        print(f"\n{task}")
        for r in rows:
            if r["task"] == task and r["a"] == "sfp" and r["b"] == "cl_sfp" \
                    and not str(r["level"]).startswith("pooled"):
                lv = r["level"] * (1 if task.startswith("pusht") else 1000)
                unit = "px" if task.startswith("pusht") else "mm"
                extra = (f"  wins {r['b_wins']}:{r['a_wins']}" if "b_wins" in r else "")
                print(f"  {lv:>6.2f} {unit}  {r['mean_a']:.2f} -> {r['mean_b']:.2f}  "
                      f"diff {r['diff']:+.3f}  p={r['p']:.3f} {star(r['p']):<3}{extra}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
