"""Compare a CL-SFP variant against the interp reference on one task (3 seeds).
Usage: python compare_variant.py <task> <variant_tag_prefix> [ah]   e.g. can _interp_swa
       ah != 8 reads perturb_ah<ah>_s100 files for the variant."""
import json, sys, numpy as np
from scipy import stats
task, var = sys.argv[1], sys.argv[2]; ah = int(sys.argv[3]) if len(sys.argv) > 3 else 8
T = ("", "_seed1", "_seed2")
BAND = {"pusht": (1.2, 2.8), "can": (0.0005, 0.00125), "square": (0.0003, 0.00125)}[task]
def ep(tag, ahs=""):
    name = f"perturb_rand{ahs}_s100_episodes.json" if task == "pusht" else f"perturb{ahs}_s100_episodes.json"
    ck = json.load(open(f"outputs/{task}/cl_sfp{tag}/eval/{name}"))["checkpoints"]
    k = next(iter(ck)); return k, {float(p): np.array(v["scores"]) for p, v in ck[k]["results"].items()}
ref = [ep("_interp" + t)[1] for t in T]
varr = [ep(var + t, "" if ah == 8 else f"_ah{ah}") for t in T]
ck = [v[0] for v in varr]; varr = [v[1] for v in varr]
lv = sorted(set.intersection(*(set(r) for r in ref + varr)))
sc = 1 if task == "pusht" else 1000
ms = lambda R, p: f"{np.mean([r[p].mean() for r in R]):.3f} ±{np.std([r[p].mean() for r in R], ddof=1):.3f}"
print(f"{task}: interp (ref) vs {var} ah{ah}  [variant ckpts {ck}]")
print(f"{'level':>7}  {'interp':>14}  {'variant':>14}  {'diff':>7}  paired p")
for p in lv:
    a = np.concatenate([r[p] for r in ref]); b = np.concatenate([r[p] for r in varr]); d = b - a
    if task == "pusht": pv = 1.0 if np.all(d == 0) else stats.wilcoxon(d, zero_method="zsplit").pvalue
    else:
        bw = int(np.sum((b > .5) & (a <= .5))); aw = int(np.sum((a > .5) & (b <= .5)))
        pv = 1.0 if aw + bw == 0 else stats.binomtest(bw, aw + bw, .5).pvalue
    print(f"{p*sc:>7.2f}  {ms(ref,p):>14}  {ms(varr,p):>14}  {d.mean():>+7.3f}  {pv:.3f}{' *' if pv < .05 else ''}")
band = [p for p in lv if BAND[0] - 1e-9 <= p <= BAND[1] + 1e-9]
a = np.concatenate([r[p] for r in ref for p in band]); b = np.concatenate([r[p] for r in varr for p in band]); d = b - a
if task == "pusht": pv = stats.wilcoxon(d, zero_method="zsplit").pvalue
else:
    bw = int(np.sum((b > .5) & (a <= .5))); aw = int(np.sum((a > .5) & (b <= .5))); pv = stats.binomtest(bw, aw + bw, .5).pvalue
print(f"static: {np.mean([r[0.0].mean() for r in ref]):.3f} -> {np.mean([r[0.0].mean() for r in varr]):.3f}   "
      f"robust({band[0]*sc:g}-{band[-1]*sc:g}): {a.mean():.3f} -> {b.mean():.3f}  diff {d.mean():+.3f}  p={pv:.1e}")
