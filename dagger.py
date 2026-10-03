"""DAgger from a Diffusion Policy teacher into CL-SFP (robomimic).

  collect : roll out a policy (student CL-SFP or the DP itself), record every raw observation
  label   : query the DP teacher at every `--stride`-th visited state -> 16-step action plan
  (train) : algo/cl_sfp.py --dagger-dir <dir> [--init-ckpt ...] mixes the labelled items with the demos

Items match RobomimicDataset: obs_seq[m] = obs_{j-1+m} (m=0..15), obs = obs_seq[:2],
action = DP plan from window (obs_{j-1}, obs_j); action[1:] is the SFP trajectory.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import algo.cl_sfp as clsfp  # noqa: E402
from env.robomimic.dagger_data import DaggerDataset  # noqa: E402,F401


def make_g(args, device):
    g = clsfp.setup(args.task, need_loader=False, need_render=False)
    if args.concat_obs0:
        g["cond_mult"] = 2
    g["width_mult"] = args.width_mult
    clsfp.CONCAT_OBS0 = args.concat_obs0
    clsfp.DIFF_OBS0 = args.diff_obs0
    return g


def collect(args):
    device = torch.device("cuda")
    g = make_g(args, device)
    env = g["env"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.policy == "dp":
        import algo.dp as dp
        nets = dp.load_ema_nets(g, args.task, args.ckpt, device, tag=args.tag)
        run = dp.rollout
    else:
        nets = clsfp.load_ema_nets(g, args.task, args.ckpt, device, tag=args.tag)
        run = clsfp.rollout

    rec, rew = [], []
    orig_reset, orig_step = env.reset, env.step

    def reset(*a, **k):
        o, info = orig_reset(*a, **k)
        rec.clear(); rew.clear(); rec.append(np.asarray(o, np.float32))
        return o, info

    def step(a):
        r = orig_step(a)
        rec.append(np.asarray(r[0], np.float32)); rew.append(float(r[1]))
        return r

    env.reset, env.step = reset, step
    levels = [float(x) for x in args.perturbs.split(",")]
    eps = [e for e in range(args.episodes) if e % args.nshards == args.shard]
    for e in eps:
        f = out / f"{args.policy}_ep{e:05d}.npz"
        if f.exists():
            continue
        seed = args.seed0 + e
        lvl = levels[e % len(levels)]
        env.seed(seed)
        score, _, steps = run(g, nets, env, seed=seed, perturb_level=lvl, device=device)
        np.savez_compressed(f, obs=np.stack(rec), rew=np.asarray(rew, np.float32), score=float(score), perturb=lvl, seed=seed)
        print(f"[{args.policy} shard{args.shard}] ep{e} seed{seed} p={lvl} score={score} steps={steps}", flush=True)


def label(args):
    import algo.dp as dp
    device = torch.device("cuda")
    g = make_g(args, device)
    obs_h, pred_h = g["obs_horizon"], g["pred_horizon"]
    stats = g["stats"]; norm = g["normalize_data"]
    net = dp.load_ema_nets(g, args.task, args.dp_ckpt, device, tag=args.dp_tag)["velocity_net"]
    sched = dp.make_scheduler(); sched.set_timesteps(dp.NUM_DIFFUSION_ITERS)
    adim = g["action"].shape[-1]

    files = sorted(Path(args.dir).glob("*_ep*.npz"))
    episodes, scores, items = [], [], []
    for f in files:
        d = np.load(f)
        o = norm(d["obs"], stats=stats["obs"]).astype(np.float32)
        N = len(o) - 1
        hit = np.flatnonzero(d["rew"] > 0.5)
        if len(hit):                                    # nothing new happens after the task is done
            N = min(N, int(hit[0]) + 16)
        ei = len(episodes)
        o = o[:N + 1]
        episodes.append(o); scores.append((f.name, float(d["score"]), float(d["perturb"])))
        for j in range(1, N - (pred_h - 2) , args.stride):      # need obs up to j+14
            items.append((ei, j))
    print(f"{len(files)} episodes, {len(items)} states to label", flush=True)

    plans = np.zeros((len(items), pred_h, adim), np.float32)
    gen = torch.Generator(device=device); gen.manual_seed(0)
    B = args.batch
    with torch.no_grad():
        for b0 in range(0, len(items), B):
            chunk = items[b0:b0 + B]
            cond = np.stack([np.concatenate([episodes[e][j - 1], episodes[e][j]]) for e, j in chunk])
            cond = torch.from_numpy(cond).to(device)
            x = torch.randn((len(chunk), pred_h, adim), device=device, generator=gen)
            for k in sched.timesteps:
                x = sched.step(net(sample=x, timestep=k, global_cond=cond), k, x).prev_sample
            plans[b0:b0 + len(chunk)] = x.cpu().numpy()
            print(f"labelled {b0 + len(chunk)}/{len(items)}", flush=True)
    np.savez(Path(args.dir) / "labels.npz", plans=plans,
             items=np.asarray(items, np.int64),
             ep_lens=np.asarray([len(e) for e in episodes], np.int64),
             obs=np.concatenate(episodes), names=np.asarray([s[0] for s in scores]))
    json.dump(scores, open(Path(args.dir) / "scores.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "label"])
    ap.add_argument("--task", default="tool_hang")
    ap.add_argument("--policy", default="student", choices=["student", "dp"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default="700")
    ap.add_argument("--concat-obs0", action="store_true")
    ap.add_argument("--diff-obs0", action="store_true")
    ap.add_argument("--width-mult", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--perturbs", default="0.0,0.4,0.8")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="dagger/r0")
    ap.add_argument("--dir", default="dagger/r0")
    ap.add_argument("--dp-ckpt", default="300")
    ap.add_argument("--dp-tag", default="")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    {"collect": collect, "label": label}[args.cmd](args)


if __name__ == "__main__":
    main()
