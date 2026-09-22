"""
sfp_1step.py - SFP with replanning every step (action horizon 1). Ablation.

Same trained weights as SFP: checkpoints are read from outputs/<task>/sfp/.
Only the rollout differs - each environment step starts a new chunk, so the
conditioning window is re-read every step just like CL-SFP, but the flow
clock is reset to t=0 each time instead of advancing with the chunk. This
isolates "fresh observation" from "time-aligned conditioning".

Results land under outputs/<task>/sfp_1step/eval/ so they sit next to sfp and
cl_sfp as a third method.

Usage
-----
    python sfp_1step.py --task pusht --mode eval --ckpt 1000 --seeds 100 --perturbs 0.0-2.0
    python sfp_1step.py --task can   --mode eval --ckpt 100-600 --seeds 100 --perturbs 0.0
    python sfp_1step.py --task pusht --mode video --ckpt 1000 --seed 0 --perturb 1.4
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import algo.sfp as sfp

METHOD = "sfp_1step"
ACTION_HORIZON = 1
WEIGHTS_FROM = "sfp"

sfp.METHOD = METHOD


def available_ckpts(task, tag=""):
    d = ROOT / "outputs" / task / f"{WEIGHTS_FROM}{tag}"
    eps = sorted(int(m.group(1)) for m in
                 (re.match(r"ep(\d+)\.ckpt$", p.name) for p in d.glob("ep*.ckpt")) if m)
    if not eps:
        raise FileNotFoundError(f"no SFP checkpoints in {d}")
    return eps


def sweep_path(task, seeds, out_json=None, action_horizon=None, tag=""):
    if out_json:
        return Path(out_json)
    return sfp.ckpt_dir(task, tag) / "eval" / f"sweep_s{seeds}.json"


_orig_rollout = sfp.rollout


def rollout(*args, **kwargs):
    kwargs["action_horizon"] = ACTION_HORIZON
    return _orig_rollout(*args, **kwargs)


_orig_load_sweep = sfp._load_sweep


def _load_sweep(*args, **kwargs):
    payload = _orig_load_sweep(*args, **kwargs)
    payload["rollout"] = METHOD
    payload["trained_by"] = WEIGHTS_FROM
    payload["method"] = METHOD
    payload["action_horizon"] = ACTION_HORIZON
    return payload


sfp.available_ckpts = available_ckpts
sfp.sweep_path = sweep_path
sfp.rollout = rollout
sfp._load_sweep = _load_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(sfp.TASKS))
    ap.add_argument("--mode", required=True, choices=["eval", "video"],
                    help="no train mode: this method shares SFP's weights "
                         "(train with sfp.py)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--perturbs", default="0.0-2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix of the SFP checkpoint dir to load from")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="steps before an episode is cut off (default: the "
                         "task's budget in env/robomimic/tasks.py)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    args.action_horizon = ACTION_HORIZON

    device = torch.device(args.device)
    g = sfp.setup(args.task, dataset_path=args.dataset, need_loader=False,
                  need_render=(args.mode == "video"))
    print("shared infra loaded; obs_dim=%d action_dim=%d  (method=%s, "
          "weights from outputs/<task>/%s, action_horizon=%d)"
          % (g["obs"].shape[-1], g["action"].shape[-1], METHOD, WEIGHTS_FROM,
             ACTION_HORIZON))

    ckpt = sfp.parse_ckpt(args.ckpt)
    if args.mode == "video":
        if ckpt is None:
            ckpt = available_ckpts(args.task, args.tag)[-1]
        if isinstance(ckpt, list):
            ckpt = ckpt[-1]
        out = args.out or (f"{args.task}_{METHOD}_ep{ckpt}"
                           f"_seed{args.seed}_p{args.perturb}.mp4")
        sfp.visualize(g, args.task, ckpt, args.seed, args.perturb, out,
                      device=device)
    else:
        if ckpt is None:
            ckpts = sfp.default_ckpts(args.task, args.tag)
        else:
            ckpts = ckpt if isinstance(ckpt, list) else [ckpt]
        print(f"evaluating checkpoint(s): {ckpts}")
        sfp.evaluate(g, args.task, args.seeds, ckpts,
                     sfp.parse_perturb(args.perturbs), args.smoke,
                     device=device, out_json=args.out_json, args=args,
                     redo=args.redo, action_horizon=ACTION_HORIZON, tag=args.tag,
                     max_steps=args.max_steps)


if __name__ == "__main__":
    main()
