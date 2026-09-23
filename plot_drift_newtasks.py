"""Drift curves for transport and tool_hang, three training seeds.

Same encoding as plot_task_curves.py (hue = method family, dash = CL-weights
variant, marker + direct label per series, band = +/-1 std over seeds). These
two tasks have no SFP-1step run, so each panel carries three series.

Usage: python plot_drift_newtasks.py   -> outputs/perf_transport_toolhang.png
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent; OUT = ROOT / "outputs"; SEEDS = ("", "_seed1", "_seed2")
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"; INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
STYLE = {"SFP": ("sfp", "#2a78d6", "-", "o"), "CL-SFP": ("cl_sfp", "#eb6834", "--", "s"),
         "CL-SFP+i": ("cl_sfp_interp", "#eb6834", "-", "D")}
NOTE = {"transport": "payload drift · 700 steps · selected ckpt · 100 eval seeds · band ±1 std",
        "tool_hang": "frame drift · 700 steps · selected ckpt · 100 eval seeds · band ±1 std"}

def runs(task, prefix):
    out = []
    for s in SEEDS:
        ck = json.load(open(OUT / task / f"{prefix}{s}" / "eval" / "perturb_s100_episodes.json"))["checkpoints"]
        res = next(iter(ck.values()))["results"]; out.append({float(p) * 1000: v["mean"] for p, v in res.items()})
    xs = sorted(set.intersection(*(set(r) for r in out))); A = np.array([[r[x] for x in xs] for r in out])
    return np.array(xs), A.mean(0), A.std(0, ddof=1), len(out)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), facecolor=SURFACE)
fig.subplots_adjust(left=0.07, right=0.93, top=0.78, bottom=0.15, wspace=0.38)
for ax, task in zip(axes, ("transport", "tool_hang")):
    ax.set_facecolor(SURFACE)
    series = [(n,) + runs(task, STYLE[n][0]) for n in STYLE]
    xmax = max(s[1][-1] for s in series); ax.set_xlim(-0.02 * xmax, xmax * 1.04)
    for n, xs, m, sd, k in series:
        _, c, ls, mk = STYLE[n]
        ax.fill_between(xs, m - sd, m + sd, color=c, alpha=0.10, linewidth=0)
        ax.plot(xs, m, color=c, linestyle=ls, linewidth=1.6, marker=mk, markersize=6.5, markerfacecolor=c,
                markeredgecolor=SURFACE, markeredgewidth=1.4, label=f"{n}  (n={k})", zorder=3)
    ends = sorted((m[-1], n, xs[-1]) for n, xs, m, sd, k in series)
    if ends[-1][0] - ends[0][0] >= 0.08:
        prev = -1.0
        for y, n, x in ends:
            yy = max(y, prev + 0.055); prev = yy
            ax.text(x + 0.025 * xmax, yy, n, va="center", ha="left", fontsize=9, color=INK2, clip_on=False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=2.4)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(AXIS); ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=AXIS)
    ax.set_ylim(-0.02, 0.62); ax.set_ylabel("success rate", color=INK2, fontsize=10)
    ax.set_xlabel("object drift (mm / step)", color=INK2, fontsize=10)
    ax.set_title(task, loc="left", color=INK, fontsize=13, fontweight="bold", pad=18)
    ax.text(0, 1.015, NOTE[task], transform=ax.transAxes, color=MUTED, fontsize=9)
fig.suptitle("transport / tool_hang under object drift - 3 training seeds", x=0.07, ha="left",
             color=INK, fontsize=14, fontweight="bold", y=0.985)
fig.text(0.07, 0.935, "hue = method family (blue SFP · orange CL-SFP);  dashed = floor-indexed CL-SFP, solid = interpolated;  "
         "y-axis capped at 0.6 - no method exceeds 0.45", color=MUTED, fontsize=9)
out = OUT / "perf_transport_toolhang.png"; fig.savefig(out, dpi=170, facecolor=SURFACE); print(f"saved -> {out}")
