"""Aggregate seeds 0/1/2: tables (mean +- std over training seeds), a paired
test pooled over seeds, and a figure with seed-mean curves.

Writes outputs/multiseed_summary.csv, outputs/multiseed_pusht.csv and
outputs/multiseed_curves.png.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
TAGS = ("", "_seed1", "_seed2")
METHODS = ("sfp", "sfp_1step", "cl_sfp")
NAMES = {"sfp": "SFP", "sfp_1step": "SFP-1step", "cl_sfp": "CL-SFP"}
COLORS = {"sfp": "#2a78d6", "sfp_1step": "#1baf7a", "cl_sfp": "#eb6834"}
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def pusht(m, tag):
    d = json.load(open(ROOT / f"outputs/pusht/{m}{tag}/eval/perturb_rand_s100_episodes.json"))
    res = d["checkpoints"]["1000"]["results"]
    return {float(p): np.asarray(v["scores"]) for p, v in res.items()}


def robomimic_best(task, m, tag):
    d = json.load(open(ROOT / f"outputs/{task}/{m}{tag}/eval/sweep_s100_episodes.json"))
    rows = {int(k): v["results"]["0.0"] for k, v in d["checkpoints"].items() if int(k) <= 600}
    ep = max(rows, key=lambda k: (rows[k]["mean"], -k))
    return ep, rows[ep]["mean"], rows


# ---------------------------------------------------------------- robomimic
lines = ["task,method,mean,std,seed0,seed1,seed2,best_ckpts"]
print("\n=== robomimic, perturb 0, best checkpoint per seed (mean +- std over 3 seeds) ===")
print(f"{'task':8}{'method':11}{'mean':>7}{'std':>7}   per-seed (best ckpt)")
for task in ("can", "lift", "square"):
    for m in METHODS:
        per = [robomimic_best(task, m, t) for t in TAGS]
        sc = np.array([p[1] for p in per])
        eps = [p[0] for p in per]
        lines.append(f"{task},{m},{sc.mean():.4f},{sc.std(ddof=1):.4f},"
                     + ",".join(f"{s:.2f}" for s in sc) + "," + "/".join(map(str, eps)))
        print(f"{task:8}{NAMES[m]:11}{sc.mean():>7.3f}{sc.std(ddof=1):>7.3f}   "
              + "  ".join(f"{s:.2f}@{e}" for s, e in zip(sc, eps)))
(ROOT / "outputs/multiseed_summary.csv").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- pusht
data = {m: [pusht(m, t) for t in TAGS] for m in METHODS}
levels = sorted(data["sfp"][0])
curves = {m: {p: np.array([np.mean(d[p]) for d in data[m]]) for p in levels} for m in METHODS}

print("\n=== Push-T (random drift), ep1000, mean +- std over 3 training seeds ===")
print(f"{'px/step':>7}  " + "  ".join(f"{NAMES[m]:>13}" for m in METHODS))
pl = ["level," + ",".join(f"{m}_mean,{m}_std" for m in METHODS)]
for p in levels:
    print(f"{p:>7.1f}  " + "  ".join(f"{curves[m][p].mean():.3f} +-{curves[m][p].std(ddof=1):.3f}" for m in METHODS))
    pl.append(f"{p}," + ",".join(f"{curves[m][p].mean():.4f},{curves[m][p].std(ddof=1):.4f}" for m in METHODS))
(ROOT / "outputs/multiseed_pusht.csv").write_text("\n".join(pl) + "\n")

print("\n=== Push-T paired test, episodes pooled over 3 seeds (n=300/level), Wilcoxon ===")


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


for a, b in (("sfp", "cl_sfp"), ("sfp", "sfp_1step"), ("sfp_1step", "cl_sfp")):
    print(f"\n  {NAMES[b]} - {NAMES[a]}")
    for p in levels:
        A = np.concatenate([d[p] for d in data[a]]); B = np.concatenate([d[p] for d in data[b]])
        d = B - A
        pv = 1.0 if np.all(d == 0) else stats.wilcoxon(d, zero_method="zsplit").pvalue
        print(f"    {p:>4.1f} px  {A.mean():.3f} -> {B.mean():.3f}  diff {d.mean():+.3f}  p={pv:.4f} {star(pv)}")
    band = [p for p in levels if 1.2 <= p <= 2.8]
    A = np.concatenate([d[p] for d in data[a] for p in band]); B = np.concatenate([d[p] for d in data[b] for p in band])
    d = B - A
    pv = stats.wilcoxon(d, zero_method="zsplit").pvalue
    print(f"    pooled 1.2-2.8 px (n={len(d)}): diff {d.mean():+.3f}  p={pv:.2e} {star(pv)}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor="#f9f9f7",
                         gridspec_kw={"width_ratios": [1.35, 1]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.8, bottom=0.16, wspace=0.25)

ax = axes[0]
for m in METHODS:
    mu = np.array([curves[m][p].mean() for p in levels]); sd = np.array([curves[m][p].std(ddof=1) for p in levels])
    ax.fill_between(levels, mu - sd, mu + sd, color=COLORS[m], alpha=0.13, linewidth=0)
    ax.plot(levels, mu, color=COLORS[m], linewidth=2, marker="o", markersize=5,
            markerfacecolor=SURFACE, markeredgewidth=2, label=NAMES[m])
ax.set_title("Push-T  ·  ep1000  ·  3 training seeds", loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Object drift (px / step, random direction per episode)", color=INK2, fontsize=9.5)
ax.set_ylabel("Success rate  (mean over seeds, band = ±1 std)", color=INK2, fontsize=9.5)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])

ax = axes[1]
tasks = ("can", "square")
w = 0.26
for i, m in enumerate(METHODS):
    mus, sds = [], []
    for task in tasks:
        sc = np.array([robomimic_best(task, m, t)[1] for t in TAGS]); mus.append(sc.mean()); sds.append(sc.std(ddof=1))
    x = np.arange(len(tasks)) + (i - 1) * w
    ax.bar(x, mus, w * 0.92, color=COLORS[m], label=NAMES[m], zorder=3)
    ax.errorbar(x, mus, yerr=sds, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for xx, mu, sd in zip(x, mus, sds):
        ax.text(xx, mu + sd + 0.025, f"{mu:.2f}", ha="center", color=INK2, fontsize=8.5)
ax.set_xticks(np.arange(len(tasks))); ax.set_xticklabels(tasks)
ax.set_title("robomimic  ·  no drift  ·  best ckpt, 3 seeds", loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("Success rate  (mean ± std over seeds)", color=INK2, fontsize=9.5)

for ax in axes:
    ax.set_facecolor(SURFACE); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

fig.suptitle("Seed robustness  —  100 episodes per point per seed", x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold", y=0.95)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=3, frameon=False, fontsize=10, labelcolor=INK2)
out = ROOT / "outputs/multiseed_curves.png"
fig.savefig(out, dpi=160)
print(f"\nsaved -> {out}")
