"""
dp.py - Diffusion Policy (DP-C, 1D UNet + DDPM) under this project's protocol.

The comparison SFP has been living next to is the number in the Diffusion
Policy paper, measured with that paper's recipe. This runs DP-C on exactly what
SFP gets here: the same dataset windows (obs_horizon 2, pred_horizon 16,
action_horizon 8), the same ConditionalUnet1D widths, the same optimizer,
schedule, EMA, batch size and epoch budget as train() in sfp.py, the same
250/700-step episodes, the same 100 eval seeds and the same checkpoint rule.
The two differences are the ones that define the method: the network denoises
the whole 16-step chunk (so its temporal convolutions are live - see the
capacity note in HANDOFF.md) and inference is 100 DDPM steps per chunk, after
which 8 actions are executed open loop, as in the paper's low-dim setting.

Checkpoints and results land in outputs/<task>/dp<tag>/, so sweep_driver,
static_summary and drift_eval work unchanged.

Usage
-----
    python dp.py --task square --mode train --epochs 1000
    python dp.py --task square --mode eval  --ckpt 100-600 --seeds 100 --perturbs 0.0
    python dp.py --task square --mode video --ckpt 300 --seed 0 --perturb 0.0008
"""

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

import algo.sfp as sfp

METHOD = "dp"
NUM_DIFFUSION_ITERS = 100        # DP-C low-dim default (paper: 100 DDPM steps)
sfp.METHOD = METHOD


def make_scheduler():
    # the settings of the official Diffusion Policy notebook / low-dim configs
    return DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_ITERS,
                         beta_schedule="squaredcos_cap_v2",
                         clip_sample=True,
                         prediction_type="epsilon")


def build_nets(g):
    """Same class and widths as SFP; Conv up/down because the input is a real
    16-step sequence here, so down/upsampling over time is meaningful again."""
    obs_dim = g["obs"].shape[-1]
    action_dim = g["action"].shape[-1]
    velocity_net = sfp.ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * g["obs_horizon"],
        updownsample_type="Conv",
        sin_embedding_scale=1,
        verbose=False,
    )
    return nn.ModuleDict({"velocity_net": velocity_net})   # key kept for load_ema_nets


def train(g, task, epochs, smoke, device=None, train_seed=0, tag=""):
    device = device or torch.device("cuda")
    dataloader = g["dataloader"]
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    ckpt_dir = ROOT / "outputs" / task / f"{METHOD}{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    nets = build_nets(g).to(device)
    print(f"params={sum(p.numel() for p in nets.parameters())/1e6:.2f}M")
    ema = EMAModel(model=nets, power=0.75)
    opt = torch.optim.AdamW(nets.parameters(), lr=1e-4, weight_decay=1e-6)
    sched = get_scheduler("cosine", optimizer=opt,
                          num_warmup_steps=len(dataloader) * 10,
                          num_training_steps=len(dataloader) * epochs)
    noise_sched = make_scheduler()

    for epoch in tqdm(range(epochs), desc=f"DP-{task}"):
        losses = []
        for nbatch in dataloader:
            nobs = nbatch["obs"].to(device)                 # (B, obs_horizon, O) chunk-start window
            naction = nbatch["action"].to(device)           # (B, pred_horizon, A)
            B = naction.shape[0]
            noise = torch.randn_like(naction)
            timesteps = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (B,), device=device).long()
            noisy = noise_sched.add_noise(naction, noise, timesteps)
            pred = nets["velocity_net"](sample=noisy, timestep=timesteps,
                                        global_cond=nobs.flatten(start_dim=1))
            loss = nn.functional.mse_loss(pred, noise)
            loss.backward(); opt.step(); opt.zero_grad(); sched.step(); ema.step(nets)
            losses.append(loss.item())
            if smoke:
                break
        if (epoch + 1) % 100 == 0 or (smoke and epoch == epochs - 1):
            torch.save({"state_dict": ema.averaged_model.state_dict(),
                        "train_seed": train_seed, "tag": tag, "method": METHOD,
                        "abs_action": g.get("abs_action", False),
                        "num_diffusion_iters": NUM_DIFFUSION_ITERS,
                        "epoch": epoch + 1, "task": task}, str(ckpt_dir / f"ep{epoch + 1}.ckpt"))
            print(f"[ep{epoch + 1}] loss={np.mean(losses):.4f}", flush=True)
        if smoke:
            print(f"[smoke] epoch {epoch} loss={np.mean(losses):.4f}")
    print(f"train done  (seed={train_seed} tag={tag!r} -> {ckpt_dir})")


def rollout(g, ema_nets, env, seed=0, perturb_level=0.0, max_steps=None,
            action_horizon=8, save_vis=False, device=None):
    """DP execution: at each chunk start, denoise a 16-step action sequence
    conditioned on the current observation window, execute action_horizon of
    them open loop (the paper's receding-horizon control), repeat."""
    device = device or torch.device("cuda")
    obs_horizon, pred_horizon = g["obs_horizon"], g["pred_horizon"]
    max_steps = g["max_steps"] if max_steps is None else int(max_steps)
    action_dim = g["action"].shape[-1]
    stats, normalize_data, unnormalize_data = g["stats"], g["normalize_data"], g["unnormalize_data"]
    net = ema_nets["velocity_net"]
    noise_sched = make_scheduler()
    noise_sched.set_timesteps(NUM_DIFFUSION_ITERS)

    obs, info = env.reset(seed_=seed, perturb_level=perturb_level)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render()] if save_vis else []
    rewards, done, step_idx = [], False, 0
    gen = torch.Generator(device=device); gen.manual_seed(seed)

    with torch.no_grad():
        while not done:
            cond = torch.from_numpy(normalize_data(np.stack(obs_deque), stats=stats["obs"])
                                    ).to(device, torch.float32).flatten().unsqueeze(0)
            naction = torch.randn((1, pred_horizon, action_dim), device=device, generator=gen)
            for k in noise_sched.timesteps:
                pred = net(sample=naction, timestep=k, global_cond=cond)
                naction = noise_sched.step(model_output=pred, timestep=k, sample=naction).prev_sample
            action_pred = unnormalize_data(naction[0].cpu().numpy(), stats=stats["action"])
            start = obs_horizon - 1
            for a in action_pred[start:start + action_horizon]:
                obs, reward, done, _, info = env.step(a)
                obs_deque.append(obs); rewards.append(reward)
                if save_vis:
                    imgs.append(env.render())
                step_idx += 1
                if step_idx >= max_steps:
                    done = True
                if done:
                    break
    return (max(rewards) if rewards else 0.0), imgs, step_idx


def available_ckpts(task, tag=""):
    d = ROOT / "outputs" / task / f"{METHOD}{tag}"
    eps = sorted(int(m.group(1)) for m in
                 (re.match(r"ep(\d+)\.ckpt$", p.name) for p in d.glob("ep*.ckpt")) if m)
    if not eps:
        raise FileNotFoundError(f"no DP checkpoints in {d}")
    return eps


def load_ema_nets(g, task, ckpt, device, ckpt_path=None, tag=""):
    """sfp.load_ema_nets looks under outputs/<task>/sfp<tag>/; DP's checkpoints
    live under dp<tag>/ and use DP's build_nets, so both are resolved here."""
    path = ckpt_path or (ROOT / "outputs" / task / f"{METHOD}{tag}" / f"ep{ckpt}.ckpt")
    if not Path(path).is_file():
        raise FileNotFoundError(f"no DP checkpoint: {path}")
    blob = torch.load(str(path), map_location=device, weights_only=False)
    nets = build_nets(g).to(device)
    ema = EMAModel(model=nets, power=0.75)
    ema.averaged_model.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    return ema.averaged_model.to(device).eval()


_orig_load_sweep = sfp._load_sweep


def _load_sweep(*args, **kwargs):
    payload = _orig_load_sweep(*args, **kwargs)
    payload.update(rollout=METHOD, trained_by=METHOD, method=METHOD,
                   num_diffusion_iters=NUM_DIFFUSION_ITERS)
    return payload


sfp.build_nets = build_nets
sfp.load_ema_nets = load_ema_nets
sfp.rollout = rollout
sfp.available_ckpts = available_ckpts
sfp._load_sweep = _load_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(sfp.TASKS))
    ap.add_argument("--mode", required=True, choices=["train", "eval", "video"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--perturbs", default="0.0-2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--abs-action", action="store_true")
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    g = sfp.setup(args.task, dataset_path=args.dataset, batch_size=args.batch_size,
                  num_workers=args.num_workers, need_loader=(args.mode == "train"),
                  need_render=(args.mode == "video"), abs_action=args.abs_action)
    print(f"shared infra loaded; obs_dim={g['obs'].shape[-1]} action_dim={g['action'].shape[-1]}"
          f"  (method={METHOD}, {NUM_DIFFUSION_ITERS} DDPM steps)")

    if args.mode == "train":
        train(g, args.task, args.epochs, args.smoke, device=device,
              train_seed=args.train_seed, tag=args.tag)
        return
    ckpt = sfp.parse_ckpt(args.ckpt)
    if args.mode == "video":
        if ckpt is None:
            ckpt = available_ckpts(args.task, args.tag)[-1]
        if isinstance(ckpt, list):
            ckpt = ckpt[-1]
        out = args.out or f"{args.task}_{METHOD}_ep{ckpt}_seed{args.seed}_p{args.perturb}.mp4"
        sfp.visualize(g, args.task, ckpt, args.seed, args.perturb, out, device=device, tag=args.tag)
    else:
        ckpts = sfp.default_ckpts(args.task, args.tag) if ckpt is None else (ckpt if isinstance(ckpt, list) else [ckpt])
        print(f"evaluating checkpoint(s): {ckpts}")
        sfp.evaluate(g, args.task, args.seeds, ckpts, sfp.parse_perturb(args.perturbs), args.smoke,
                     device=device, out_json=args.out_json, args=args, redo=args.redo,
                     action_horizon=args.action_horizon, tag=args.tag, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
