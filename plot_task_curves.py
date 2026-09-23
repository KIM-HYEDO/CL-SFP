"""Success curves for can, lift and square, three training seeds each.

can / square: drift curves - success at the statically selected checkpoint
against object drift (mm/step), for the four methods that have one.
lift: no drift curve was ever run (every method saturates at 1.000), so its
panel is the checkpoint sweep at drift 0 - the only measurement that exists.

Encoding follows final_summary.py: hue names the method family (SFP blue,
1-step aqua, CL-SFP orange), dash marks the CL-weights variant, and every series
also carries its own marker and a direct label, so identity never rests on
color alone. Bands are +/- 1 std over training seeds.

Usage: python plot_task_curves.py          -> outputs/perf_can_lift_square.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEEDS = ("", "_seed1", "_seed2")

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
# name -> (dir prefix, color, linestyle, marker)
STYLE = {
    "SFP":          ("sfp",           "#2a78d6", "-",  "o"),
    "SFP-1step":    ("sfp_1step",     "#1baf7a", "-",  "^"),
    "CL-SFP":       ("cl_sfp",        "#eb6834", "--", "s"),
    "CL-SFP+i":     ("cl_sfp_interp", "#eb6834", "-",  "D"),
    "CL-SFP-1step": ("cl_sfp_1step",  "#1baf7a", "--", "v"),
}
DRIFT_METHODS = ("SFP", "SFP-1step", "CL-SFP", "CL-SFP+i")
STATIC_METHODS = ("SFP", "SFP-1step", "CL-SFP", "CL-SFP-1step")


def drift_runs(task, prefix):
    runs = []
    for s in SEEDS:
        f = OUT / task / f"{prefix}{s}" / "eval" / "perturb_s100_episodes.json"
        if not f.is_file():
            continue
        ck = json.load(open(f))["checkpoints"]
        res = next(iter(ck.values()))["results"]
        runs.append({float(p) * 1000: v["mean"] for p, v in res.items()})   # m -> mm
    return runs


def sweep_runs(task, prefix):
    runs = []
    for s in SEEDS:
        f = OUT / task / f"{prefix}{s}" / "eval" / "sweep_s100_episodes.json"
        if not f.is_file():
            continue
        ck = json.load(open(f))["checkpoints"]
        runs.append({int(k): v["results"]["0.0"]["mean"]
                     for k, v in ck.items() if v["results"].get("0.0")})
    return runs


def aggregate(runs):
    """Mean/std over seeds on the x values every seed measured."""
    xs = sorted(set.intersection(*(set(r) for r in runs)))
    arr = np.array([[r[x] for x in xs] for r in runs])
    return np.array(xs), arr.mean(0), (arr.std(0, ddof=1) if len(runs) > 1 else np.zeros(len(xs))), len(runs)


def style_axes(ax, title, subtitle, xlabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=AXIS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("success rate", color=INK2, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=18)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9)


def draw(ax, series, xlabel, legend_loc="upper right"):
    """series: list of (name, xs, mean, std, n_seeds)."""
    for name, xs, m, sd, n in series:
        _, color, ls, marker = STYLE[name]
        ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.10, linewidth=0)
        ax.plot(xs, m, color=color, linestyle=ls, linewidth=1.6, marker=marker,
                markersize=6.5, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.4, label=f"{name}  (n={n})", zorder=3)
    # Direct labels at the right end, spread apart in data units when they
    # collide. When every series converges on one value there is nothing a
    # label could point at - the legend carries identity (marks-and-anatomy).
    ends = sorted((m[-1], name, xs[-1]) for name, xs, m, sd, n in series)
    if ends[-1][0] - ends[0][0] >= 0.08:
        gap, prev = 0.055, -1.0
        x_pad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
        for y, name, x in ends:
            yy = max(y, prev + gap); prev = yy
            ax.text(x + x_pad, yy, name, va="center", ha="left", fontsize=9,
                    color=INK2, clip_on=False)
    ax.legend(loc=legend_loc, frameon=False, fontsize=8.5, labelcolor=INK2,
              handlelength=2.4, borderaxespad=0.2)


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), facecolor=SURFACE)
fig.subplots_adjust(left=0.05, right=0.955, top=0.78, bottom=0.15, wspace=0.42)

for ax, task in zip(axes[:2], ("can", "square")):
    series = []
    for name in DRIFT_METHODS:
        runs = drift_runs(task, STYLE[name][0])
        if runs:
            series.append((name,) + aggregate(runs))
    ax.set_xlim(-0.03, 1.56)
    draw(ax, series, "object drift (mm / step)")
    style_axes(ax, task, "success vs drift at the selected checkpoint · 100 eval seeds · band = ±1 std",
               "object drift (mm / step)")

ax = axes[2]
series = []
for name in STATIC_METHODS:
    runs = sweep_runs("lift", STYLE[name][0])
    if runs:
        series.append((name,) + aggregate(runs))
draw(ax, series, "training epoch", legend_loc="lower right")
style_axes(ax, "lift", "no drift run - all methods saturate at 1.0; static checkpoint sweep",
           "training epoch")
ax.set_xticks(sorted(set(int(x) for s in series for x in s[1])))

fig.suptitle("SFP vs closed-loop variants - 3 training seeds", x=0.05, ha="left",
             color=INK, fontsize=14, fontweight="bold", y=0.985)
fig.text(0.05, 0.935, "hue = method family (blue SFP · aqua 1-step · orange CL-SFP);  dashed = CL-SFP weights variant;  "
         "250-step episodes, not robomimic's 400", color=MUTED, fontsize=9)
out = OUT / "perf_can_lift_square.png"
fig.savefig(out, dpi=170, facecolor=SURFACE)
print(f"saved -> {out}")
