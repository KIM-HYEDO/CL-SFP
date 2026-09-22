"""Final aggregation: drift curves (3 seeds where available) for every method
on Push-T / can / square, a static-vs-robust Pareto table, and one figure.

Writes outputs/final_curves.csv, outputs/final_pareto.csv, outputs/final_figure.png.
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
STYLE = {  # name -> (color, linestyle)
    "SFP":        ("#2a78d6", "-"),
    "SFP-ah4":    ("#2a78d6", "--"),
    "SFP-ah2":    ("#2a78d6", ":"),
    "SFP-1step":  ("#1baf7a", "-"),
    "CL-SFP":     ("#eb6834", "--"),
    "CL-SFP+i":   ("#eb6834", "-"),
}
ROBUST_BAND = {"pusht": (1.2, 2.8), "can": (0.0005, 0.00125), "square": (0.0003, 0.00125)}


def _scores(path):
    """{level: mean} from an episodes json holding a single checkpoint."""
    ck = json.load(open(path))["checkpoints"]
    res = next(iter(ck.values()))["results"]
    return {float(p): v["mean"] for p, v in res.items()}


def pusht_sources():
    f = lambda m, t, name="perturb_rand_s100_episodes.json": ROOT / f"outputs/pusht/{m}{t}/eval/{name}"
    return {
        "SFP":       [f("sfp", t) for t in SEEDS],
        "SFP-ah4":   [f("sfp", t, "perturb_rand_ah4_s100_episodes.json") for t in SEEDS],
        "SFP-ah2":   [f("sfp", t, "perturb_rand_ah2_s100_episodes.json") for t in SEEDS],
        "SFP-1step": [f("sfp_1step", t) for t in SEEDS],
        "CL-SFP":    [f("cl_sfp", t) for t in SEEDS],
        "CL-SFP+i":  [f("cl_sfp", "_interp" + t) for t in SEEDS],
    }


def robomimic_sources(task):
    f = lambda m, t, name="perturb_s100_episodes.json": ROOT / f"outputs/{task}/{m}{t}/eval/{name}"
    return {
        "SFP":       [f("sfp", t) for t in SEEDS],
        "SFP-ah4":   [f("sfp", t, "perturb_ah4_s100_episodes.json") for t in SEEDS],
        "SFP-ah2":   [f("sfp", "", "perturb_ah2_s100_episodes.json")],
        "SFP-1step": [f("sfp_1step", t) for t in SEEDS],
        "CL-SFP":    [f("cl_sfp", t) for t in SEEDS],
        "CL-SFP+i":  [f("cl_sfp", "_interp" + t) for t in SEEDS],
    }


def load(sources, xmax=None):
    out = {}
    for name, paths in sources.items():
        runs = [_scores(p) for p in paths if p.is_file()]
        if not runs:
            continue
        levels = sorted(set.intersection(*(set(r) for r in runs)))
        if xmax is not None:
            levels = [l for l in levels if l <= xmax + 1e-12]
        arr = np.array([[r[l] for l in levels] for r in runs])  # (seeds, levels)
        out[name] = (levels, arr)
    return out


def pareto(curves, band):
    rows = []
    for name, (levels, arr) in curves.items():
        lv = np.array(levels)
        static = arr[:, lv == 0.0].mean(axis=1)
        mask = (lv >= band[0] - 1e-12) & (lv <= band[1] + 1e-12)
        robust = arr[:, mask].mean(axis=1)
        rows.append((name, static.mean(), static.std(ddof=1) if len(static) > 1 else 0.0,
                     robust.mean(), robust.std(ddof=1) if len(robust) > 1 else 0.0, arr.shape[0]))
    return rows


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def main():
    tasks = {"pusht": load(pusht_sources()),
             "can": load(robomimic_sources("can"), xmax=0.0015),
             "square": load(robomimic_sources("square"), xmax=0.0015)}
    scale = {"pusht": 1.0, "can": 1000.0, "square": 1000.0}
    unit = {"pusht": "px/step", "can": "mm/step", "square": "mm/step"}

    # ---------------------------------------------------------------- tables
    with open(OUT / "final_curves.csv", "w") as fh:
        fh.write("task,method,n_seeds,level,mean,std\n")
        for task, curves in tasks.items():
            for name, (levels, arr) in curves.items():
                for j, l in enumerate(levels):
                    sd = arr[:, j].std(ddof=1) if arr.shape[0] > 1 else 0.0
                    fh.write(f"{task},{name},{arr.shape[0]},{l * scale[task]:g},{arr[:, j].mean():.4f},{sd:.4f}\n")

    print("=== Pareto: static = success at drift 0; robust = mean success over the mid band ===")
    with open(OUT / "final_pareto.csv", "w") as fh:
        fh.write("task,method,n_seeds,static_mean,static_std,robust_mean,robust_std\n")
        for task, curves in tasks.items():
            b = ROBUST_BAND[task]
            print(f"\n{task}  (robust band {b[0] * scale[task]:g}-{b[1] * scale[task]:g} {unit[task]})")
            print(f"  {'method':10}{'seeds':>6}{'static':>16}{'robust':>16}")
            for name, sm, ss, rm, rs, n in pareto(curves, b):
                fh.write(f"{task},{name},{n},{sm:.4f},{ss:.4f},{rm:.4f},{rs:.4f}\n")
                print(f"  {name:10}{n:>6}   {sm:.3f} ±{ss:.3f}   {rm:.3f} ±{rs:.3f}")

    print("\n=== Drift curves (mean over seeds) ===")
    for task, curves in tasks.items():
        names = list(curves)
        levels = curves["SFP"][0]
        print(f"\n{task} ({unit[task]})  " + "".join(f"{n:>11}" for n in names))
        for j, l in enumerate(levels):
            row = f"{l * scale[task]:>8.2f}     "
            for n in names:
                lv, arr = curves[n]
                row += f"{arr[:, lv.index(l)].mean():>11.3f}" if l in lv else f"{'-':>11}"
            print(row)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), facecolor="#f9f9f7",
                             gridspec_kw={"height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.07, wspace=0.24, hspace=0.42)
    titles = {"pusht": "Push-T  ·  ep1000", "can": "can  ·  best ckpt", "square": "square  ·  best ckpt"}
    for col, task in enumerate(("pusht", "can", "square")):
        curves = tasks[task]
        ax = axes[0, col]
        for name in STYLE:
            if name not in curves:
                continue
            lv, arr = curves[name]
            x = np.array(lv) * scale[task]
            mu = arr.mean(axis=0)
            c, ls = STYLE[name]
            if arr.shape[0] > 1:
                sd = arr.std(axis=0, ddof=1)
                ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.08, linewidth=0)
            ax.plot(x, mu, color=c, linestyle=ls, linewidth=2 if ls == "-" else 1.6,
                    marker="o", markersize=4, markerfacecolor=SURFACE, markeredgewidth=1.6,
                    label=f"{name} ({arr.shape[0]}s)")
        style(ax, titles[task], f"Object drift ({unit[task]}, random direction per episode)")
        ax.set_ylim(0, 1.05)
        if col == 0:
            ax.set_ylabel("Success rate  (mean over seeds, band ±1 std)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")

        ax = axes[1, col]
        b = ROBUST_BAND[task]
        rows = pareto(curves, b)
        xs = [r[1] for r in rows]; ys = [r[3] for r in rows]
        xr = (max(xs) - min(xs)) or 1; yr = (max(ys) - min(ys)) or 1
        placed = []
        for name, sm, ss, rm, rs, n in sorted(rows, key=lambda r: -r[3]):
            c, ls = STYLE[name]
            ax.errorbar(sm, rm, xerr=ss, yerr=rs, fmt="o", color=c, markersize=8,
                        markerfacecolor=c if ls == "-" else SURFACE, markeredgewidth=2,
                        elinewidth=1, capsize=2.5, zorder=3)
            # labels: up-right by default; drop below when a placed label is close
            dy = 4
            for px, py in placed:
                if abs(sm - px) / xr < 0.18 and abs(rm - py) / yr < 0.12:
                    dy = -12
            ax.annotate(name, (sm, rm), xytext=(7, dy), textcoords="offset points",
                        color=INK2, fontsize=8.5)
            placed.append((sm, rm))
        style(ax, f"{task}  ·  static vs robust",
              "Success at zero drift")
        ax.set_ylabel(f"Mean success, drift {b[0] * scale[task]:g}-{b[1] * scale[task]:g} {unit[task]}",
                      color=INK2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)

    fig.suptitle("Closing the loop in a streaming flow policy  —  100 episodes per point per seed",
                 x=0.05, ha="left", color=INK, fontsize=13, fontweight="bold", y=0.965)
    fig.savefig(OUT / "final_figure.png", dpi=150)
    print(f"\nsaved -> {OUT / 'final_curves.csv'}\nsaved -> {OUT / 'final_pareto.csv'}\nsaved -> {OUT / 'final_figure.png'}")


if __name__ == "__main__":
    main()
