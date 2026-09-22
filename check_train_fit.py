"""Did training learn the field? Integrate the trained flow on dataset windows.

Feeds recorded observation windows to a trained SFP checkpoint, starts the flow
on the demonstrated action at t=0 and takes the 8 Euler steps the rollout would
take, then compares each step with the recorded action. Reported against the
"stay at a0" baseline, so the ratio says how much of the demonstrated motion the
network reproduces, in normalised units, independent of task difficulty.

Same ratio across tasks means the network fits every task equally well, and a
gap in success rate is closed-loop execution (horizon, tolerance, memory), not
a training or inference bug. Measured 2026-09-22: can 0.46, square 0.59,
tool_hang 0.46, transport 0.52.

Usage: python check_train_fit.py <task> <epoch>      e.g. tool_hang 500
"""

import sys, numpy as np, torch
ROOT = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import algo.sfp as sfp
task, ckpt = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
g = sfp.setup(task, need_loader=True, batch_size=512, num_workers=0)
net = sfp.load_ema_nets(g, task, ckpt, dev)
H, P = g["obs_horizon"], g["pred_horizon"]; T = P - H; dt = 1.0 / T
ds = g["dataset"]; rng = np.random.default_rng(0)
idx = rng.choice(len(ds), size=min(2048, len(ds)), replace=False)
batch = torch.utils.data.default_collate([ds[int(i)] for i in idx])
obs = batch["obs"].to(dev).flatten(1)                 # SFP: chunk-start window
xi = batch["action"][:, H - 1:, :].to(dev)            # (B, 15, A) normalized
B, _, A = xi.shape
na = xi[:, 0:1, :].clone()                            # start on the demo, t = 0
mse_model, mse_zero = [], []
with torch.no_grad():
    for i in range(8):
        t = torch.full((B,), i * dt, device=dev)
        v = net["velocity_net"](sample=na, timestep=t, global_cond=obs)
        na = na + v * dt
        tgt = xi[:, i + 1, :]
        mse_model.append(((na[:, 0] - tgt) ** 2).mean().item())
        mse_zero.append(((xi[:, 0] - tgt) ** 2).mean().item())   # "stay at a0" baseline
m, z = np.array(mse_model), np.array(mse_zero)
print(f"[{task} ep{ckpt}] windows={B}  action_dim={A}")
print(f"[{task}]  step:      " + " ".join(f"{i+1:>6d}" for i in range(8)))
print(f"[{task}]  model RMSE " + " ".join(f"{np.sqrt(x):6.3f}" for x in m))
print(f"[{task}]  stay  RMSE " + " ".join(f"{np.sqrt(x):6.3f}" for x in z))
print(f"[{task}]  error ratio model/stay (8-step mean): {np.sqrt(m.mean()/z.mean()):.2f}   (<<1 = learned the field)")
