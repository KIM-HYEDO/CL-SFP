"""
cl_sfp.py - Closed-Loop Streaming Flow Policy (CL-SFP) on Push-T and robomimic.

Learning code only. Simulators and datasets live under env/:
    env/pusht/pusht_env.py     env/pusht/pusht_data.py       task: pusht
    env/robomimic/env.py       env/robomimic/data.py         tasks: can, lift, square

Both families expose the same contract (Push-T style env API, obs/obs_seq/action
batches), so everything in this file is task-agnostic.

Usage
-----
    python cl_sfp.py --task pusht  --mode train --epochs 1000
    python cl_sfp.py --task can    --mode train --epochs 1000
    python cl_sfp.py --task pusht  --mode eval  --ckpt 1000 --seeds 100 --perturbs 0.0-2.0
    python cl_sfp.py --task square --mode video --ckpt 1000 --seed 0

--perturb displaces the manipulated object every step: pixels/step on Push-T,
metres/step on robomimic (see env/robomimic/env.py). It is the disturbance that
a chunk-start observation goes stale against, so it is the axis on which CL-SFP
is supposed to separate from SFP.
"""

import argparse
import collections
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

# Repository root, so this script runs from any working directory
# (`python exp/sfp.py`, `python -m exp.sfp`, or from inside exp/ itself).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# diffusion-policy helpers
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# Environments and datasets are loaded per task inside setup(); see the TASKS
# registry below. Importing robomimic pulls in robosuite/MuJoCo, so it stays
# lazy - a pusht run must not pay for it.

try:
    from streaming_flow_policy.all import set_random_seed
except Exception:  # keep the script runnable without the package installed
    def set_random_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_random_seed(0)


# =============================================================================
# 1. Neural network architectures
# =============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        self.dim = dim
        self.scale = scale  # added - SFP

    def forward(self, x):
        x = x * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConvUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LinearDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)  # flatten spatial dimensions
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)
        return x


class LinearUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)  # flatten spatial dimensions
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)
        return x


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x    : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim ]
        returns out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 updownsample_type: Literal['Conv', 'Linear'],  # added for SFP
                 sin_embedding_scale,                           # added for SFP
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8,
                 verbose=True,
                 ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM, in
          addition to the diffusion step embedding (usually obs_horizon*obs_dim).
        diffusion_step_embed_dim: Size of positional encoding for step k.
        down_dims: Channel size for each UNet level; length = number of levels.
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, scale=sin_embedding_scale),  # added - SFP
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if updownsample_type == 'Linear':  # added for SFP
                downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            elif updownsample_type == 'Conv':
                downsample_layer = ConvDownsample1d(dim_out) if not is_last else nn.Identity()
            else:
                raise ValueError(f"Unsupported updownsample_type: {updownsample_type}")
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            if updownsample_type == 'Linear':  # added for SFP
                upsample_layer = LinearUpsample1d(dim_in) if not is_last else nn.Identity()
            elif updownsample_type == 'Conv':
                upsample_layer = ConvUpsample1d(dim_in) if not is_last else nn.Identity()
            else:
                raise ValueError(f"Unsupported updownsample_type: {updownsample_type}")
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                upsample_layer,
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        if verbose:
            print("Number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())))

    def forward(self,
                sample: Tensor,
                timestep: Union[Tensor, float, int],
                global_cond=None,
                ) -> Tensor:
        """
        sample: (B, T, input_dim)
        timestep: (B,) or scalar
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T) -> (B,T,C)
        x = x.moveaxis(-1, -2)
        return x


# =============================================================================
# 2. Conditional flow matching (CFM) inputs and targets
# =============================================================================
def LinearlyInterpolateTrajectory(ξ, t):
    """
    Vectorized positions/velocities of each trajectory in a batch at given
    times, using linear interpolation.

    ξ (Tensor, shape=(B, T, A)): batch of action trajectories.
    t (Tensor, shape=(B,)): batch of times in [0, 1].

    Returns:
        ξt   (Tensor, shape=(B, A)): positions at time t
        dξdt (Tensor, shape=(B, A)): velocities at time t
    """
    B, T, A = ξ.shape

    # Lower/upper limits of the bins the time-points lie in.
    scaled_t = t * (T - 1)                       # (B,) in [0, T-1]
    l = scaled_t.floor().long().clamp(0, T - 2)  # (B,) lower bin limits
    u = (l + 1).clamp(0, T - 1)                  # (B,) upper bin limits
    λ = scaled_t - l.float()                     # fractional part in [0, 1]

    # Query the values at the upper and lower bin limits.
    batch_idx = torch.arange(B, device=ξ.device)  # (B,)
    ξl = ξ[batch_idx, l, :]  # (B, A)
    ξu = ξ[batch_idx, u, :]  # (B, A)

    # Linearly interpolate between bin limits to get position.
    λ = λ.unsqueeze(-1)      # (B, 1)
    ξt = ξl + λ * (ξu - ξl)  # (B, A)

    # Velocity as first-order hold; bin interval is Δt = 1 / (T-1).
    dξdt = (ξu - ξl) * (T - 1)  # (B, A)

    return ξt, dξdt


def SampleCFMInputsAndTargets(ξt, dξdt, t, k, σ0, σ_min=0.0):
    """
    Sample inputs/targets for the conditional flow matching loss:
        a ~ N(ξ(t), σ₀² exp(-2kt))   (Eq. 3 in the paper)
        v = -k (a - ξ(t)) + dξdt(t)  (Eq. 2 in the paper)

    σ_min > 0 floors the *sampling* std at σ_min while keeping the target
    field unchanged. The field -k(a - ξ) + ξ' is defined for every a; the
    floor only regresses it on a wider support late in the flow, where
    σ(t) would otherwise shrink below the deviations a disturbed rollout
    actually produces.

    Returns:
        a (Tensor, shape=(B, A)): noised actions at time t
        v (Tensor, shape=(B, A)): noised action velocity targets at time t
    """
    t = t.unsqueeze(-1)  # (B, 1)
    std = torch.clamp(σ0 * torch.exp(-k * t), min=σ_min)
    sampled_error = std * torch.randn_like(ξt)  # (B, A)
    a = ξt + sampled_error        # (B, A) ⟸ Eq. 3
    v = -k * sampled_error + dξdt  # (B, A) ⟸ Eq. 2
    return a, v


# =============================================================================
# 3. Shared setup (replaces `exec_shared` on the notebook)
# =============================================================================
# Which tasks exist, what each one observes, and how long an episode may run
# all live in one table, because they stopped agreeing once the suite grew past
# the three single-arm tasks. Importing it is cheap: tasks.py pulls in nothing.
from env.robomimic.tasks import MAX_STEPS, ROBOMIMIC_TASKS  # noqa: E402

DEFAULT_DATASET = {
    "pusht": str(ROOT / "env/pusht/data/pusht_cchi_v7_replay.zarr.zip"),
    **{t: str(ROOT / f"env/robomimic/data/{t}/low_dim.hdf5")
       for t in ROBOMIMIC_TASKS},
}


TASKS = ("pusht",) + ROBOMIMIC_TASKS


def _build_pusht(dataset_path, pred_horizon, obs_horizon, action_horizon,
                 env_seed, perturb_level, need_render, task="pusht", abs_action=False):
    from env.pusht.pusht_env import PushTEnv
    from env.pusht.pusht_data import (PushTDataset, make_dataloader,
                                      normalize_data, unnormalize_data)

    env = PushTEnv()
    env.seed(env_seed)
    dataset = PushTDataset(dataset_path=dataset_path,
                           pred_horizon=pred_horizon,
                           obs_horizon=obs_horizon,
                           action_horizon=action_horizon)
    # Push-T actions are gripper positions, so the flow starts at where the
    # gripper already is.
    def initial_action(obs, action_dim):
        return np.asarray(obs[:action_dim], dtype=np.float32)

    return (env, dataset, make_dataloader, normalize_data, unnormalize_data,
            initial_action)


def _build_robomimic(dataset_path, pred_horizon, obs_horizon, action_horizon,
                     env_seed, perturb_level, need_render, task=None,
                     abs_action=False):
    from env.robomimic.env import make_env
    from env.robomimic.data import (RobomimicDataset, make_dataloader,
                                    normalize_data, unnormalize_data)
    from env.robomimic.tasks import gripper_dims, obs_keys

    # An offscreen EGL context is only needed to render frames, and creating
    # one prints MuJoCo/EGL teardown noise at exit, so skip it otherwise.
    # The simulator and the dataset must concatenate the same keys in the same
    # order, so both are handed the one list rather than each taking a default.
    keys = obs_keys(task)
    env = make_env(dataset_path, obs_keys=keys, perturb_level=perturb_level,
                   render_offscreen=need_render, task=task, abs_action=abs_action)
    env.seed(env_seed)
    dataset = RobomimicDataset(dataset_path=dataset_path,
                               pred_horizon=pred_horizon,
                               obs_horizon=obs_horizon,
                               action_horizon=action_horizon,
                               obs_keys=keys,
                               gripper_dims=gripper_dims(task, abs_action),
                               abs_action=abs_action)
    if abs_action:
        # absolute pose: "stay put" is the pose the arm is in right now
        def initial_action(obs, action_dim):
            return env.current_eef_action()
    else:
        # robomimic delta actions: "stay put" is the zero vector rather than a
        # slice of the observation.
        def initial_action(obs, action_dim):
            return np.zeros(action_dim, dtype=np.float32)

    return (env, dataset, make_dataloader, normalize_data, unnormalize_data,
            initial_action)


def setup(task, dataset_path=None, batch_size=1024, num_workers=1,
          need_loader=True, env_seed=500, perturb_level=0.0,
          need_render=False, abs_action=False):
    """Build env + dataset + dataloader and return them in a namespace dict.

    Both task families expose the same contract, so everything downstream of
    this function is task-agnostic:
      * env speaks the Push-T API (reset -> (obs, info), step -> 5-tuple)
      * dataset items carry obs / obs_seq / action
      * g["initial_action"](obs, action_dim) gives the flow's starting action
    """
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}; expected one of {TASKS}")
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET[task]
        if abs_action:
            # the converted copy; convert_abs_actions.py writes it
            dataset_path = dataset_path.replace("low_dim.hdf5", "low_dim_abs.hdf5")

    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    builder = _build_pusht if task == "pusht" else _build_robomimic
    (env, dataset, make_dataloader, normalize_data, unnormalize_data,
     initial_action) = builder(dataset_path, pred_horizon, obs_horizon,
                               action_horizon, env_seed, perturb_level,
                               need_render, task, abs_action)

    # One probe step, so obs/action shapes come from the live env.
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    dataloader = None
    if need_loader:
        dataloader = make_dataloader(dataset, batch_size=batch_size,
                                     num_workers=num_workers)

    return {
        "task": task,
        "env": env,
        "obs": obs,
        "action": action,
        "dataset": dataset,
        "dataloader": dataloader,
        "stats": dataset.stats,
        "pred_horizon": pred_horizon,
        "obs_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "max_steps": MAX_STEPS[task],
        "abs_action": abs_action,
        "normalize_data": normalize_data,
        "unnormalize_data": unnormalize_data,
        "initial_action": initial_action,
    }


# =============================================================================
# 4. Model
# =============================================================================
def build_nets(g):
    obs_dim = g["obs"].shape[-1]
    action_dim = g["action"].shape[-1]
    obs_horizon = g["obs_horizon"]
    cond_obs = obs_dim * obs_horizon

    velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=cond_obs,   # anchor
        updownsample_type="Linear",
        sin_embedding_scale=100,
    )

    return nn.ModuleDict({"velocity_net": velocity_net})


def load_ema_nets(g, task, ckpt, device, ckpt_path=None, tag=""):
    """Rebuild the model and load an EMA checkpoint into eval mode.

    ckpt_path, when given, names the file directly. SFP and CL-SFP share this
    architecture exactly and differ only in which observation window the field
    is conditioned on, so an SFP checkpoint can be scored under the CL-SFP
    rollout and vice versa.
    """
    if ckpt_path:
        path = str(ckpt_path)
    else:
        # current layout, then the older flat names, so checkpoints trained
        # before the outputs/<task>/cl_sfp/ split are still loadable.
        out_dir = ROOT / "outputs" / task
        candidates = [out_dir / f"cl_sfp{tag}" / f"ep{ckpt}.ckpt",
                      out_dir / f"cl_sfp_ep{ckpt}.ckpt",
                      out_dir / f"{task}_state_ep{ckpt}.ckpt"]
        for cand in candidates:
            if cand.is_file():
                path = str(cand)
                break
        else:
            raise FileNotFoundError(
                f"no checkpoint for epoch {ckpt}; looked for "
                + ", ".join(str(c) for c in candidates))
    blob = torch.load(path, map_location=device, weights_only=False)
    nets = build_nets(g).to(device)
    ema = EMAModel(model=nets, power=0.75)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    ema.averaged_model.load_state_dict(sd)
    return ema.averaged_model.to(device).eval()


def ckpt_meta(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return {k: v for k, v in blob.items() if k != "state_dict"}
    return {}


# =============================================================================
# 5. Training
# =============================================================================
def train(g, task, epochs, smoke, device=None, train_seed=0, tag="",
          cond_interp=False, sigma_min=0.0):
    device = device or torch.device("cuda")
    dataloader = g["dataloader"]
    obs_horizon = g["obs_horizon"]
    pred_horizon = g["pred_horizon"]

    sigma0, k = 0.4, 10
    # Re-seed here so --train-seed controls init and data order regardless of
    # the module-level set_random_seed(0) that ran at import.
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    ckpt_dir = ROOT / "outputs" / task / f"cl_sfp{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    nets = build_nets(g).to(device)
    print(f"params={sum(p.numel() for p in nets.parameters())/1e6:.2f}M")
    ema = EMAModel(model=nets, power=0.75)
    opt = torch.optim.AdamW(nets.parameters(), lr=1e-4, weight_decay=1e-6)
    sched = get_scheduler("cosine", optimizer=opt,
                          num_warmup_steps=len(dataloader) * 10,
                          num_training_steps=len(dataloader) * epochs)
    T = pred_horizon - obs_horizon

    for epoch in tqdm(range(epochs), desc=f"CL-SFP-{task}"):
        losses = []
        for nbatch in dataloader:
            nobs_seq = nbatch["obs_seq"].to(device)
            naction = nbatch["action"].to(device)
            xi = naction[:, obs_horizon - 1:, :]
            B = xi.shape[0]
            t = torch.rand(B, device=device)

            xit, dxidt = LinearlyInterpolateTrajectory(xi, t)
            a, v = SampleCFMInputsAndTargets(xit, dxidt, t, k, sigma0, sigma_min)
            a, v = a.unsqueeze(1), v.unsqueeze(1)

            # This is the whole difference from SFP. SFP conditions every
            # integration step of a chunk on the chunk-start window; here the
            # window advances with flow time, so the field sees the world as it
            # is when each action is commanded.
            #
            # Alignment follows the official SFP implementation, which pairs the
            # conditioning with the instant the action is commanded from: it
            # starts the flow at a0 = nobs[-1, :2] and conditions on that same
            # nobs. CL-SFP holds that invariant at every flow time rather than
            # only at the chunk start.
            #
            # Concretely: xi[i] is action index i+obs_horizon-1, commanded from
            # the observation at that same index, and the window
            # nobs_seq[i : i+obs_horizon] ends exactly there. rollout() reads
            # its window at the matching point, before env.step.
            b_idx = torch.arange(B, device=device).unsqueeze(1)
            if cond_interp:
                # The target xi(t) slides linearly from xi[i] to xi[i+1] inside
                # bin i, so the window slides with it: at fractional time
                # i+lam the conditioning is lerp(window_i, window_{i+1}, lam).
                # With the floor-indexed window below, t just under a bin edge
                # pairs a window one step older with the same target as t just
                # over it, so the field gets conflicting supervision at the
                # very grid times the rollout queries.
                scaled = t * T
                i0 = scaled.floor().long().clamp(0, T - 1)
                lam = (scaled - i0.float()).clamp(0, 1)[:, None, None]
                idx0 = i0.unsqueeze(1) + torch.arange(obs_horizon, device=device)
                win = (1 - lam) * nobs_seq[b_idx, idx0] + lam * nobs_seq[b_idx, idx0 + 1]
            else:
                t_idx = (t * T).long().clamp(0, T)
                idx = t_idx.unsqueeze(1) + torch.arange(obs_horizon, device=device)
                win = nobs_seq[b_idx, idx]                          # (B, H, O)
            current_flat = win.flatten(start_dim=1)                 # (B, H*O)

            vhat = nets["velocity_net"](sample=a, timestep=t,
                                        global_cond=current_flat)
            loss = nn.functional.mse_loss(v, vhat)

            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            ema.step(nets)
            losses.append(loss.item())
            if smoke:
                break

        if (epoch + 1) % 100 == 0 or (smoke and epoch == epochs - 1):
            path = str(ckpt_dir / f"ep{epoch + 1}.ckpt")
            torch.save({"state_dict": ema.averaged_model.state_dict(),
                        "train_seed": train_seed, "tag": tag,
                        "abs_action": g.get("abs_action", False),
                        "cond_interp": cond_interp, "sigma_min": sigma_min,
                        "epoch": epoch + 1, "task": task}, path)
            print(f"[ep{epoch + 1}] loss={np.mean(losses):.4f}", flush=True)
        if smoke:
            print(f"[smoke] epoch {epoch} loss={np.mean(losses):.4f}")
    print(f"train done  (seed={train_seed} tag={tag!r} -> {ckpt_dir})")


# =============================================================================
# 6. Closed-loop inference
# =============================================================================
def rollout(g, ema_nets, env, seed=0, perturb_level=0.0, max_steps=None,
            action_horizon=8, save_vis=False, device=None):
    """Run one episode. Returns (score, frames, steps).

    CL-SFP execution: the conditioning window is re-read after every executed
    step, so the field integrates against the current world rather than the
    chunk-start snapshot. Everything else matches SFP: one Euler step per
    environment step, and the flow state carries across the chunk boundary
    while the flow clock restarts at 0.
    """
    device = device or torch.device("cuda")
    obs_horizon = g["obs_horizon"]
    pred_horizon = g["pred_horizon"]
    action_dim = g["action"].shape[-1]
    # The episode budget belongs to the task, not to this function: 250 steps
    # is plenty for can/lift/square but cuts transport and tool_hang off before
    # the task is even reachable. See env/robomimic/tasks.py MAX_STEPS.
    max_steps = g["max_steps"] if max_steps is None else int(max_steps)
    stats = g["stats"]
    normalize_data = g["normalize_data"]
    unnormalize_data = g["unnormalize_data"]

    obs, info = env.reset(seed_=seed, perturb_level=perturb_level)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render()] if save_vis else []
    rewards, done, step_idx = [], False, 0

    a0 = g["initial_action"](obs, action_dim)
    na = torch.from_numpy(normalize_data(a0, stats=stats["action"])).to(
        device, torch.float32)
    na_prev = na.unsqueeze(0).unsqueeze(0)

    dt = 1.0 / (pred_horizon - obs_horizon)

    while not done:
        na = na_prev
        with torch.no_grad():
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

                ncur = normalize_data(np.stack(obs_deque), stats=stats["obs"])
                current_flat = torch.from_numpy(ncur).to(
                    device, torch.float32).flatten().unsqueeze(0)
                
                t = torch.tensor(i * dt, device=device, dtype=torch.float32)
                nv = ema_nets["velocity_net"](sample=na, timestep=t,
                                              global_cond=current_flat)
                na = na + nv * dt
        na_prev = na.detach()

    return (max(rewards) if rewards else 0.0), imgs, step_idx


# =============================================================================
# 7. Evaluation
# =============================================================================
# The two scripts are identical from here down apart from this name.
METHOD = "cl_sfp"


def ckpt_dir(task, tag=""):
    return ROOT / "outputs" / task / f"{METHOD}{tag}"


def available_ckpts(task, tag=""):
    """Epoch numbers of every ep*.ckpt on disk, ascending."""
    d = ckpt_dir(task, tag)
    eps = sorted(int(m.group(1)) for m in
                 (re.match(r"ep(\d+)\.ckpt$", p.name) for p in d.glob("ep*.ckpt")) if m)
    if not eps:
        raise FileNotFoundError(f"no checkpoints in {d}")
    return eps


def default_ckpts(task, tag=""):
    """What `--ckpt auto` means, and it differs by task family.

    Push-T reports a disturbance curve for one model: take the last checkpoint
    and vary the perturbation. Robomimic reports a training curve: sweep every
    checkpoint and give each one mean score.
    """
    eps = available_ckpts(task, tag)
    return [eps[-1]] if task == "pusht" else eps


def sweep_path(task, seeds, out_json=None, action_horizon=None, tag=""):
    """One file per (task, method, seed count) - every checkpoint goes in it.

    A non-default execution chunk length gets its own file: chunk=1 and chunk=8
    are different policies at inference and must not be averaged together.
    """
    if out_json:
        return Path(out_json)
    ahtag = "" if action_horizon in (None, 8) else f"_ah{action_horizon}"
    return ckpt_dir(task, tag) / "eval" / f"sweep_s{seeds}{ahtag}.json"


def episodes_path(path):
    """Where the per-seed episode records go, out of the way of the result."""
    return path.with_name(path.stem + "_episodes.json")


def _load_sweep(path, task, seeds, g, tag=""):
    """Open an existing sweep, or start a fresh payload.

    Resume reads the episodes file, because `path` itself carries only the
    table and has nothing to resume from.
    """
    ep = episodes_path(path)
    if ep.is_file():
        try:
            payload = json.load(open(ep))
            if payload.get("checkpoints"):
                return payload
        except (json.JSONDecodeError, OSError) as exc:
            print(f"could not reuse {ep} ({exc}); starting a new sweep")
    return {
        # `rollout` is this script's execution scheme; `trained_by` is the
        # method whose checkpoint is scored. They differ under --ckpt-path
        # (e.g. an SFP checkpoint executed with the CL-SFP rollout).
        "rollout": METHOD,
        "trained_by": METHOD,
        "method": METHOD,   # kept for older readers; equals trained_by
        "tag": tag,
        "task": task,
        "seeds": seeds,
        "action_horizon": g["action_horizon"],
        "max_steps": g["max_steps"],
        "abs_action": g.get("abs_action", False),
        # which object the drift displaces (robomimic only); the result is a
        # different experiment for a different object, so it travels with it
        "perturb_object": getattr(g["env"], "perturb_object", None),
        "pred_horizon": g["pred_horizon"],
        "obs_horizon": g["obs_horizon"],
        "obs_dim": int(g["obs"].shape[-1]),
        "action_dim": int(g["action"].shape[-1]),
        "checkpoints": {},
    }


def _record_eval_horizon(payload, action_horizon, max_steps=None):
    payload["eval_action_horizon"] = int(action_horizon)
    if max_steps is not None:
        payload["max_steps"] = int(max_steps)


def _stamp(payload, args):
    """Refresh the bookkeeping fields derived from the checkpoint entries."""
    payload["perturbs"] = sorted({float(p) for e in
                                  payload["checkpoints"].values()
                                  for p in e.get("results", {})})
    payload["updated"] = datetime.now().isoformat(timespec="seconds")
    if args is not None:
        payload["args"] = {kk: vv for kk, vv in vars(args).items()}


def _score_of(entry, perturbs):
    """Selection metric: mean of the per-perturb means over the levels asked for.

    With a single level this is just that level's score. Averaging keeps a
    checkpoint from winning on the static case alone when the sweep also
    covers perturbed ones.
    """
    res = entry.get("results", {})
    vals = [res[str(p)]["mean"] for p in perturbs if str(p) in res]
    return float(np.mean(vals)) if vals else float("nan")


def _flush(payload, path):
    """Two files: the table you read, and the per-episode record you don't.

    Per-seed scores stay on disk because two runs share the seed list, so they
    are what allows a paired SFP vs CL-SFP comparison later. They just do not
    belong in the file you open to look at a result.
    """
    with open(episodes_path(path), "w") as fh:
        json.dump(payload, fh, indent=2)
    slim = {k: v for k, v in payload.items() if k != "checkpoints"}
    with open(path, "w") as fh:
        json.dump(slim, fh, indent=2)


def _write_csv(path, cols, table):
    """The same table the run prints, ready to plot."""
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in table:
            fh.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    return csv_path


def report_perturb_curve(payload, path, task, seeds, ckpt, perturbs):
    """Push-T: one checkpoint, one row per perturbation level."""
    res = payload["checkpoints"].get(str(ckpt), {}).get("results", {})
    table = [{"perturb": float(p), "score": res[str(p)]["mean"],
              "ci95": res[str(p)]["ci95"], "n": res[str(p)]["n"]}
             for p in perturbs if str(p) in res]
    payload["kind"] = "perturb_curve"
    payload["ckpt"] = int(ckpt)
    payload["table"] = table
    _flush(payload, path)
    csv_path = _write_csv(path, ["perturb", "score", "ci95", "n"], table)
    print()
    print(f"  {task}  {payload['method']}  ep{ckpt}  ({seeds} seeds)")
    print(f"  {'perturb':>8}  {'score':>8}  {'95% CI':>9}")
    print("  " + "-" * 31)
    for r in table:
        print(f"  {r['perturb']:>8.1f}  {r['score']:>8.4f}  {r['ci95']:>9.4f}")
    return csv_path


def report_ckpt_sweep(payload, path, task, seeds, perturbs):
    """Robomimic: every checkpoint, one mean score each."""
    table = []
    for c, e in sorted(payload["checkpoints"].items(), key=lambda kv: int(kv[0])):
        s = _score_of(e, perturbs)
        if s != s:      # NaN: nothing measured for this checkpoint
            continue
        res = e.get("results", {})
        ci = [res[str(p)]["ci95"] for p in perturbs if str(p) in res]
        table.append({"ckpt": int(c), "score": s,
                      "ci95": float(np.mean(ci)) if ci else float("nan")})
    metric = f"mean over perturbs {[float(p) for p in perturbs]}"
    payload["kind"] = "ckpt_sweep"
    payload["metric"] = metric
    payload["table"] = table
    _flush(payload, path)
    csv_path = _write_csv(path, ["ckpt", "score", "ci95"], table)
    print()
    print(f"  {task}  {payload['method']}  ({seeds} seeds, {metric})")
    print(f"  {'ckpt':>6}  {'score':>8}  {'95% CI':>9}")
    print("  " + "-" * 29)
    for r in table:
        print(f"  {r['ckpt']:>6}  {r['score']:>8.4f}  {r['ci95']:>9.4f}")
    return csv_path


def eval_one(g, task, seeds, ckpt, perturbs, smoke, device=None,
             action_horizon=8, ckpt_path=None, tag="", on_result=None,
             max_steps=None):
    """Score one checkpoint over the perturbation levels. No file writing here;
    `on_result(perturb_key, entry)` is called after each level so the caller can
    checkpoint partial progress.

    Per-seed scores are kept, not just the mean: two runs share the seed list,
    so the individual episodes are what allow a later paired comparison.
    """
    device = device or torch.device("cuda")
    env = g["env"]
    ema_nets = load_ema_nets(g, task, ckpt, device, ckpt_path=ckpt_path, tag=tag)

    results = {}
    for perturb in perturbs:
        scores, steps = [], []
        for s in tqdm(range(seeds), desc=f"ep{ckpt} perturb{perturb}",
                      leave=False):
            env.seed(s)
            score, _, nsteps = rollout(g, ema_nets, env, seed=s,
                               perturb_level=perturb, device=device,
                               action_horizon=action_horizon,
                               max_steps=max_steps)
            scores.append(float(score)); steps.append(int(nsteps))
        arr = np.asarray(scores, dtype=np.float64)
        n = len(arr)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        results[str(perturb)] = {"mean": mean, "std": std, "sem": sem,
                                 "ci95": 1.96 * sem, "n": n, "scores": scores,
                                 "steps": steps, "steps_mean": float(np.mean(steps))}
        print(f"[{task}] ep{ckpt} perturb{perturb}  (n={n})  "
              f"score={mean:.4f} +/- {1.96 * sem:.4f} (95% CI)  std={std:.4f}",
              flush=True)
        if on_result is not None:
            on_result(str(perturb), results[str(perturb)])
        if smoke:
            break
    return results


def evaluate(g, task, seeds, ckpts, perturbs, smoke, device=None,
             out_json=None, args=None, redo=False, action_horizon=8,
             ckpt_path=None, tag="", max_steps=None):
    """Evaluate one or more checkpoints into a single sweep file.

    The file is rewritten after every checkpoint, so an interrupted sweep keeps
    what it has already measured and a later run picks up where it stopped.
    """
    if not isinstance(ckpts, (list, tuple)):
        ckpts = [ckpts]

    max_steps = g["max_steps"] if max_steps is None else int(max_steps)
    path = sweep_path(task, seeds, out_json, action_horizon, tag)
    if max_steps != g["max_steps"] and not out_json:
        # A shorter or longer episode is a different measurement of the same
        # checkpoint, so it gets its own file rather than overwriting the sweep
        # every existing result on this task was measured into.
        path = path.with_name(f"{path.stem}_ms{max_steps}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_sweep(path, task, seeds, g, tag)
    if ckpt_path:
        payload["checkpoints"] = {}   # a foreign checkpoint: start clean
    _record_eval_horizon(payload, action_horizon, max_steps)
    if ckpt_path:
        payload["ckpt_path"] = str(ckpt_path)
        # infer the training method from outputs/<task>/<method>/... if possible
        m = re.search(r"outputs/[^/]+/(sfp|cl_sfp)/", str(ckpt_path).replace("\\", "/"))
        payload["trained_by"] = m.group(1) if m else "unknown"
        payload["method"] = payload["trained_by"]

    for ckpt in ckpts:
        key = str(ckpt)
        have = payload["checkpoints"].get(key, {}).get("results", {})
        if not redo and all(str(p) in have for p in perturbs):
            print(f"[{task}] ep{ckpt}: already measured, skipping "
                  f"(--redo to force)")
            continue
        # Resume at perturbation granularity: measure only the missing levels and
        # write the file after each one, so an interrupted sweep loses at most one
        # level instead of the whole checkpoint.
        todo = list(perturbs) if redo else [p for p in perturbs if str(p) not in have]
        if len(todo) < len(perturbs):
            print(f"[{task}] ep{ckpt}: resuming, {len(perturbs) - len(todo)} level(s) already measured")
        merged = dict(have)

        def _save(pk, entry, _key=key, _merged=merged):
            _merged[pk] = entry
            payload["checkpoints"][_key] = {
                "results": dict(_merged),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "argv": sys.argv[1:],
            }
            _stamp(payload, args)
            _flush(payload, path)

        results = eval_one(g, task, seeds, ckpt, todo, smoke, device=device,
                           action_horizon=action_horizon, ckpt_path=ckpt_path, tag=tag,
                           on_result=_save, max_steps=max_steps)
        merged.update(results)
        payload["checkpoints"][key] = {
            "results": merged,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv[1:],
        }
        _stamp(payload, args)
        _flush(payload, path)

    # --- report -----------------------------------------------------------
    # Also run when every checkpoint was skipped, so a pure resume still emits
    # the table.
    _stamp(payload, args)
    if task == "pusht":
        csv_path = report_perturb_curve(payload, path, task, seeds, ckpts[-1], perturbs)
    else:
        csv_path = report_ckpt_sweep(payload, path, task, seeds, perturbs)
    print()
    print(f"saved -> {path}")
    print(f"saved -> {csv_path}")
    print(f"         per-episode records in {episodes_path(path).name}")
    return payload


def visualize(g, task, ckpt, seed, perturb, out, device=None, tag=""):
    """Run a single episode and save it as a video (mp4, gif fallback)."""
    device = device or torch.device("cuda")
    env = g["env"]
    # tag selects the checkpoint directory exactly as in eval; without it a
    # video of "cl_sfp --tag _interp" would silently show the plain cl_sfp model
    ema_nets = load_ema_nets(g, task, ckpt, device, tag=tag)

    env.seed(seed)
    score, imgs, _ = rollout(g, ema_nets, env, seed=seed, perturb_level=perturb,
                          save_vis=True, device=device)
    print(f"[{task}] seed={seed} perturb={perturb} "
          f"score={score:.4f} frames={len(imgs)}")

    frames = [im.astype(np.uint8) for im in imgs]
    try:
        import imageio
        imageio.mimsave(out, frames, fps=20)
    except Exception as e:
        from PIL import Image
        out = os.path.splitext(out)[0] + ".gif"
        pil = [Image.fromarray(im) for im in frames]
        pil[0].save(out, save_all=True, append_images=pil[1:],
                    duration=50, loop=0)
        print(f"(mp4 save failed: {e}; fell back to gif)")
    print(f"saved video -> {out}")
    return out


# =============================================================================
# 7. CLI
# =============================================================================
def parse_ckpt(spec):
    """'auto' -> None (per-task default) | '1000' -> 1000
    | '100-500' -> [100,200,...,500] | '100,300' -> [100,300]"""
    if spec == "auto":
        return None
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1, 100))
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    return int(spec)


def parse_perturb(spec):
    """'0.0-2.0' -> [0.0, 0.2, ..., 2.0] | '0.0,0.6' -> [0.0, 0.6]"""
    if "-" in spec:
        lo, hi = map(float, spec.split("-"))
        step = 0.2
        n = round((hi - lo) / step)
        return [round(lo + i * step, 10) for i in range(n + 1)]
    return [float(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--mode", required=True, choices=["train", "eval", "video"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--ckpt", default="auto",
                    help="auto: last checkpoint on pusht, every checkpoint on "
                         "robomimic. Or an epoch, a '100-1000' range, or a list.")
    ap.add_argument("--perturbs", default="0.0-2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None,
                    help="path to the zarr replay buffer")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="training RNG seed (init + data order)")
    ap.add_argument("--tag", default="",
                    help="suffix on the checkpoint dir, e.g. _seed1 -> outputs/<task>/cl_sfp_seed1/")
    ap.add_argument("--sigma-min", type=float, default=0.0,
                    help="train: floor on the CFM sampling std (see SampleCFMInputsAndTargets)")
    ap.add_argument("--cond-interp", action="store_true",
                    help="train: interpolate the conditioning window between "
                         "grid steps so it stays aligned with the target xi(t) "
                         "at every continuous t (see train())")
    ap.add_argument("--ckpt-path", default=None,
                    help="load this checkpoint file instead of outputs/<task>/<method>/. "
                         "Requires --out-json so the result does not land in a sweep file.")
    ap.add_argument("--action-horizon", type=int, default=8,
                    help="actions executed per chunk at eval time; the model "
                         "is trained for 8. 1 = replan every step.")
    ap.add_argument("--abs-action", action="store_true",
                    help="robomimic: absolute end-effector actions (pos + 6D "
                         "rotation + gripper) from low_dim_abs.hdf5, controller "
                         "in control_delta=False. Use a distinct --tag (e.g. _abs).")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="steps before an episode is cut off (default: the "
                         "task's budget in env/robomimic/tasks.py). A "
                         "non-default value gets its own sweep file.")
    ap.add_argument("--redo", action="store_true",
                    help="re-measure checkpoints already in the sweep file")
    ap.add_argument("--out-json", default=None,
                    help="where to write eval results "
                         "(default outputs/<task>/cl_sfp/eval/sweep_s<seeds>.json)")
    args = ap.parse_args()
    if args.ckpt_path and not args.out_json:
        ap.error("--ckpt-path needs --out-json")

    device = torch.device(args.device)
    g = setup(args.task,
              dataset_path=args.dataset,
              batch_size=args.batch_size,
              num_workers=args.num_workers,
              need_loader=(args.mode == "train"),
              need_render=(args.mode == "video"),
              abs_action=args.abs_action)
    print("shared infra loaded; obs_dim=%d action_dim=%d"
          % (g["obs"].shape[-1], g["action"].shape[-1]))

    if args.mode == "train":
        train(g, args.task, args.epochs, args.smoke, device=device,
              train_seed=args.train_seed, tag=args.tag,
              cond_interp=args.cond_interp, sigma_min=args.sigma_min)
    elif args.mode == "video":
        ckpt = parse_ckpt(args.ckpt)
        if ckpt is None:
            ckpt = available_ckpts(args.task, args.tag)[-1]
        if isinstance(ckpt, list):
            ckpt = ckpt[-1]
        out = args.out or (f"{args.task}_cl_sfp_ep{ckpt}"
                           f"_seed{args.seed}_p{args.perturb}.mp4")
        visualize(g, args.task, ckpt, args.seed, args.perturb, out,
                  device=device, tag=args.tag)
    else:
        ckpt = parse_ckpt(args.ckpt)
        if ckpt is None:
            ckpts = default_ckpts(args.task, args.tag)
        else:
            ckpts = ckpt if isinstance(ckpt, list) else [ckpt]
        print(f"evaluating checkpoint(s): {ckpts}")
        evaluate(g, args.task, args.seeds, ckpts, parse_perturb(args.perturbs),
                 args.smoke, device=device, out_json=args.out_json,
                 args=args, redo=args.redo, action_horizon=args.action_horizon,
                 ckpt_path=args.ckpt_path, tag=args.tag,
                 max_steps=args.max_steps)


if __name__ == "__main__":
    main()