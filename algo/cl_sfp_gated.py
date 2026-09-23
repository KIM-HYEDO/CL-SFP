"""
cl_sfp_gated.py - CL-SFP with a surprise-rewound flow clock.

Same trained weights as CL-SFP (outputs/<task>/cl_sfp<tag>/); only the
rollout differs. CL-SFP advances the flow clock t by dt every step and
re-reads the observation, so late in a chunk the field is queried at small
sigma(t) = sigma0*exp(-k t) - it expects the running action to sit almost
exactly on the trajectory implied by the *current* observation. When a fresh
observation changes that trajectory (a disturbance), the running action is
suddenly far off relative to sigma(t) and the field is out of distribution.
SFP-1step avoids this by always querying at t=0 (sigma0), at the cost of
precision when nothing happened.

Here the clock is rewound just enough to cover the change. At every step the
field is queried twice at the same (a, t): with the previous window and with
the current one. The flow-matching target is v = -k (a - xi(t)) + xi'(t), so
for a fixed (a, t) the difference between the two predictions is, up to the
xi' term, k times the shift of the implied trajectory:

    delta_xi ~= || v(a,t | obs_new) - v(a,t | obs_old) || / k

The clock is then set to the time whose noise level matches that shift,

    t <- min(t, t*),   sigma(t*) = gamma * delta_xi

so t is untouched when the observation confirmed the plan, and falls toward 0
(the SFP-1step regime) when it overturned it. gamma=1 is the literal
noise-matching rule; larger gamma rewinds more aggressively. With
--gate hard the rule is a threshold instead: t <- 0 iff delta_xi > tau.

Usage
-----
    python cl_sfp_gated.py --task pusht --mode eval  --ckpt 1000 --seeds 100 --perturbs 0.0-2.0 --gamma 5
    python cl_sfp_gated.py --task pusht --mode calib --ckpt 1000 --gammas 1,5,20 --perturbs 0.0,1.4,2.4
Results land under outputs/<task>/cl_sfp_gated_g<gamma><tag>/eval/.
"""

import argparse
import atexit
import collections
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import algo.cl_sfp as base

WEIGHTS_FROM = "cl_sfp"
SIGMA0, K = 0.4, 10.0          # must match train() in cl_sfp.py

GATE = {"kind": "soft", "gamma": 1.0, "tau": 0.0, "tfix": 0.0, "snap": True}
DT = 1.0 / 14
STATS = {"t_used": [], "rewound": [], "n": 0}


def _method_name():
    k = GATE["kind"]
    if k == "soft":
        return f"cl_sfp_gated_g{GATE['gamma']:g}"
    if k == "excess":
        return f"cl_sfp_excess_g{GATE['gamma']:g}"
    if k == "fixed":
        return f"cl_sfp_t{GATE['tfix']:g}"
    return f"cl_sfp_gated_tau{GATE['tau']:g}"


def rewind_time(t, delta_xi, progress=0.0):
    """Clock time whose noise level covers the observed plan shift.

    `progress` is the shift expected from ordinary advance along the
    trajectory (|xi'| dt); the "excess" gate only reacts to what exceeds it.
    """
    if GATE["kind"] == "hard":
        return 0.0 if delta_xi > GATE["tau"] else t
    if GATE["kind"] == "excess":
        delta_xi = max(0.0, delta_xi - progress)
    s = GATE["gamma"] * delta_xi
    if s <= 0:
        return t
    t_star = math.log(SIGMA0 / s) / K
    if GATE["snap"]:
        # The field was trained with the window indexed by floor(t*T), so it
        # is only consistent at grid times; land on the grid, never between.
        t_star = math.floor(t_star / DT + 1e-9) * DT
    return max(0.0, min(t, t_star))


def rollout(g, ema_nets, env, seed=0, perturb_level=0.0, max_steps=None,
            action_horizon=8, save_vis=False, device=None):
    device = device or torch.device("cuda")
    obs_horizon, pred_horizon = g["obs_horizon"], g["pred_horizon"]
    action_dim = g["action"].shape[-1]
    # The episode budget belongs to the task, not to this function: 250 steps
    # is plenty for can/lift/square but cuts transport and tool_hang off before
    # the task is even reachable. See env/robomimic/tasks.py MAX_STEPS.
    max_steps = g["max_steps"] if max_steps is None else int(max_steps)
    stats = g["stats"]
    normalize_data, unnormalize_data = g["normalize_data"], g["unnormalize_data"]
    net = ema_nets["velocity_net"]

    obs, info = env.reset(seed_=seed, perturb_level=perturb_level)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render()] if save_vis else []
    rewards, done, step_idx = [], False, 0

    a0 = g["initial_action"](obs, action_dim)
    na = torch.from_numpy(normalize_data(a0, stats=stats["action"])).to(
        device, torch.float32)[None, None]
    dt = 1.0 / (pred_horizon - obs_horizon)

    def window():
        ncur = normalize_data(np.stack(obs_deque), stats=stats["obs"])
        return torch.from_numpy(ncur).to(device, torch.float32).flatten()[None]

    prev_cond = None
    t_used, rewound = [], []
    with torch.no_grad():
        while not done:
            for i in range(action_horizon):
                a = unnormalize_data(na.cpu().numpy().squeeze(axis=(0, 1)),
                                     stats=stats["action"])
                obs, reward, done, _, info = env.step(a)
                obs_deque.append(obs)
                rewards.append(reward)
                if save_vis:
                    imgs.append(env.render())
                step_idx += 1
                if step_idx >= max_steps:
                    done = True
                if done:
                    break

                cond = window()
                t = GATE["tfix"] if GATE["kind"] == "fixed" else i * dt
                tt = torch.tensor(t, device=device, dtype=torch.float32)
                nv = net(sample=na, timestep=tt, global_cond=cond)
                if GATE["kind"] != "fixed" and prev_cond is not None and t > 0:
                    nv_old = net(sample=na, timestep=tt, global_cond=prev_cond)
                    delta_xi = float(torch.linalg.norm(nv - nv_old)) / K
                    progress = float(torch.linalg.norm(nv)) * dt
                    t_new = rewind_time(t, delta_xi, progress)
                    if t_new < t - 1e-9:
                        tt = torch.tensor(t_new, device=device, dtype=torch.float32)
                        nv = net(sample=na, timestep=tt, global_cond=cond)
                        rewound.append(1.0)
                        t = t_new
                    else:
                        rewound.append(0.0)
                t_used.append(t)
                na = na + nv * dt
                prev_cond = cond

    STATS["t_used"].append(float(np.mean(t_used)) if t_used else 0.0)
    STATS["rewound"].append(float(np.mean(rewound)) if rewound else 0.0)
    STATS["n"] += 1
    return (max(rewards) if rewards else 0.0), imgs, step_idx


def _report_stats():
    if STATS["n"]:
        print(f"[gate {_method_name()}] episodes={STATS['n']}  "
              f"mean t used={np.mean(STATS['t_used']):.3f} (plain CL-SFP: 0.250)  "
              f"rewind rate={np.mean(STATS['rewound']):.3f}", flush=True)


atexit.register(_report_stats)


def available_ckpts(task, tag=""):
    d = ROOT / "outputs" / task / f"{WEIGHTS_FROM}{tag}"
    eps = sorted(int(m.group(1)) for m in
                 (re.match(r"ep(\d+)\.ckpt$", p.name) for p in d.glob("ep*.ckpt")) if m)
    if not eps:
        raise FileNotFoundError(f"no CL-SFP checkpoints in {d}")
    return eps


def sweep_path(task, seeds, out_json=None, action_horizon=None, tag=""):
    if out_json:
        return Path(out_json)
    return base.ckpt_dir(task, tag) / "eval" / f"sweep_s{seeds}.json"


_orig_load_sweep = base._load_sweep


def _load_sweep(*args, **kwargs):
    payload = _orig_load_sweep(*args, **kwargs)
    payload.update(rollout=_method_name(), trained_by=WEIGHTS_FROM,
                   method=_method_name(), gate=dict(GATE))
    return payload


base.available_ckpts = available_ckpts
base.sweep_path = sweep_path
base.rollout = rollout
base._load_sweep = _load_sweep


def calibrate(g, task, ckpt, settings, perturbs, seeds, seed_from, device, tag):
    """Score the gate on held-out episode seeds for several gammas.

    Seeds start at `seed_from` so the choice of gamma never touches the
    0..seeds-1 episodes used by --mode eval. gamma=0 disables the gate
    (plain CL-SFP) and is always included as the reference.
    """
    ema_nets = base.load_ema_nets(g, task, ckpt, device, tag=tag)
    env = g["env"]
    print(f"\ncalibration on seeds {seed_from}..{seed_from + seeds - 1}, ep{ckpt}")
    print(f"{'setting':>18} " + " ".join(f"{'p=' + str(p):>10}" for p in perturbs)
          + f"   {'mean':>6}   {'<t>':>5}  {'rewind':>6}")
    for kind, val in [("soft", 0.0)] + list(settings):
        GATE.update(kind=kind, gamma=val if kind != "fixed" else 1.0,
                    tfix=val if kind == "fixed" else 0.0)
        STATS.update(t_used=[], rewound=[], n=0)
        means = []
        for p in perturbs:
            sc = []
            for s in range(seed_from, seed_from + seeds):
                env.seed(s)
                score, _, _ = rollout(g, ema_nets, env, seed=s, perturb_level=p,
                                      device=device)
                sc.append(float(score))
            means.append(np.mean(sc))
        name = "CL-SFP" if (kind, val) == ("soft", 0.0) else f"{kind} {val:g}"
        print(f"{name:>18} " + " ".join(f"{m:>10.3f}" for m in means)
              + f"   {np.mean(means):>6.3f}   {np.mean(STATS['t_used']):>5.3f}"
              f"  {np.mean(STATS['rewound']):>6.3f}", flush=True)
    STATS.update(t_used=[], rewound=[], n=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(base.TASKS))
    ap.add_argument("--mode", required=True, choices=["eval", "calib", "video"])
    ap.add_argument("--gate", default="soft", choices=["soft", "hard", "excess", "fixed"])
    ap.add_argument("--tfix", type=float, default=0.0, help="fixed: constant flow time")
    ap.add_argument("--no-snap", action="store_true", help="do not snap rewound t to the grid")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--calib", default="soft:1,soft:10",
                    help="calib: comma list of kind:value, e.g. fixed:0.1,excess:10")
    ap.add_argument("--seed-from", type=int, default=100, help="calib: first episode seed")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--perturbs", default="0.0-2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="steps before an episode is cut off (default: the "
                         "task's budget in env/robomimic/tasks.py)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    GATE.update(kind=args.gate, gamma=args.gamma, tau=args.tau, tfix=args.tfix,
                snap=not args.no_snap)
    base.METHOD = _method_name()
    args.action_horizon = 8

    device = torch.device(args.device)
    g = base.setup(args.task, dataset_path=args.dataset, need_loader=False,
                   need_render=(args.mode == "video"))
    print(f"shared infra loaded; obs_dim={g['obs'].shape[-1]} action_dim="
          f"{g['action'].shape[-1]}  (method={base.METHOD}, weights from "
          f"outputs/<task>/{WEIGHTS_FROM}{args.tag})")

    ckpt = base.parse_ckpt(args.ckpt)
    if ckpt is None:
        ckpt = available_ckpts(args.task, args.tag)[-1] if args.task == "pusht" \
            else base.default_ckpts(args.task, args.tag)
    if args.mode == "calib":
        ck = ckpt[-1] if isinstance(ckpt, list) else ckpt
        settings = [(kv.split(":")[0], float(kv.split(":")[1])) for kv in args.calib.split(",")]
        calibrate(g, args.task, ck, settings,
                  base.parse_perturb(args.perturbs), args.seeds, args.seed_from,
                  device, args.tag)
    elif args.mode == "video":
        ck = ckpt[-1] if isinstance(ckpt, list) else ckpt
        out = args.out or f"{args.task}_{base.METHOD}_ep{ck}_seed{args.seed}_p{args.perturb}.mp4"
        base.visualize(g, args.task, ck, args.seed, args.perturb, out, device=device, tag=args.tag)
    else:
        ckpts = ckpt if isinstance(ckpt, list) else [ckpt]
        print(f"evaluating checkpoint(s): {ckpts}")
        base.evaluate(g, args.task, args.seeds, ckpts, base.parse_perturb(args.perturbs),
                      args.smoke, device=device, out_json=args.out_json, args=args,
                      redo=args.redo, action_horizon=8, tag=args.tag,
                      max_steps=args.max_steps)


if __name__ == "__main__":
    main()
