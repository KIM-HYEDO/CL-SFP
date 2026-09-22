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
    python cl_sfp.py --task pusht  --mode eval  --ckpt 1000 --seeds 50 --perturbs 0.0-2.0
    python cl_sfp.py --task square --mode video --ckpt 1000 --seed 0

Note on --perturb for robomimic: Push-T displaces the block mid-chunk, whereas
robomimic perturbation is Gaussian noise on the command (see env/robomimic/env.py).
Those measure different things; do not put them on one axis.
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
from exp.contractive import ContractiveField  # noqa: E402  (ROOT is on sys.path above)

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


def SampleCFMInputsAndTargets(ξt, dξdt, t, k, σ0, σ_const=None, σ_min=None):
    """
    Sample inputs/targets for the conditional flow matching loss:
        a ~ N(ξ(t), σ₀² exp(-2kt))   (Eq. 3 in the paper)
        v = -k (a - ξ(t)) + dξdt(t)  (Eq. 2 in the paper)

    Returns:
        a (Tensor, shape=(B, A)): noised actions at time t
        v (Tensor, shape=(B, A)): noised action velocity targets at time t
    """
    t = t.unsqueeze(-1)  # (B, 1)
    if σ_const is not None:
        # Constant training spread. The CFM path's σ₀e^{-kt} collapses to ~0 past
        # t≈0.4, leaving no lever arm to learn ∂v/∂a = -k there; the network
        # then reads position off h instead of a. A constant spread keeps the
        # contraction learnable at every t. The target below is unchanged, so
        # the learned drift is the same field, sampled on a wider tube.
        sampled_error = σ_const * torch.randn_like(ξt)
    elif σ_min is not None:
        # Floored spread: the CFM schedule early on, a fixed tube later. Keeps the
        # contraction learnable at late t (samples exist off the path) while the
        # early precision of the schedule is untouched.
        σt = torch.clamp(σ0 * torch.exp(-k * t), min=σ_min)
        sampled_error = σt * torch.randn_like(ξt)
    else:
        sampled_error = σ0 * torch.exp(-k * t) * torch.randn_like(ξt)  # (B, A)
    a = ξt + sampled_error        # (B, A) ⟸ Eq. 3
    v = -k * sampled_error + dξdt  # (B, A) ⟸ Eq. 2
    return a, v


# =============================================================================
# 3. Shared setup (replaces `exec_shared` on the notebook)
# =============================================================================
DEFAULT_DATASET = {
    "pusht": str(ROOT / "env/pusht/data/pusht_cchi_v7_replay.zarr.zip"),
    **{t: str(ROOT / f"env/robomimic/data/{t}/low_dim.hdf5")
       for t in ("can", "lift", "square")},
}


ROBOMIMIC_TASKS = ("can", "lift", "square")
TASKS = ("pusht",) + ROBOMIMIC_TASKS


def _build_pusht(dataset_path, pred_horizon, obs_horizon, action_horizon,
                 env_seed, perturb_level, need_render):
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

    # Layout for drift augmentation (raw units: px). obs = agent_pos(2) block_pos(2) angle(1);
    # action = agent target position (absolute).
    drift_spec = {"obj_pos": [2, 3], "robot_pos": [0, 1], "act_pos": [0, 1],
                  "act_mode": "abs", "rel": None, "lift": None, "unit": "px"}
    return (env, dataset, make_dataloader, normalize_data, unnormalize_data,
            initial_action, drift_spec)


def _build_robomimic(dataset_path, pred_horizon, obs_horizon, action_horizon,
                     env_seed, perturb_level, need_render):
    from env.robomimic.env import make_env
    from env.robomimic.data import (RobomimicDataset, make_dataloader,
                                    normalize_data, unnormalize_data)

    # An offscreen EGL context is only needed to render frames, and creating
    # one prints MuJoCo/EGL teardown noise at exit, so skip it otherwise.
    env = make_env(dataset_path, perturb_level=perturb_level,
                   render_offscreen=need_render)
    env.seed(env_seed)
    dataset = RobomimicDataset(dataset_path=dataset_path,
                               pred_horizon=pred_horizon,
                               obs_horizon=obs_horizon,
                               action_horizon=action_horizon)
    # robomimic actions are end-effector deltas, not positions, so "stay put"
    # is the zero vector rather than a slice of the observation.
    def initial_action(obs, action_dim):
        return np.zeros(action_dim, dtype=np.float32)

    # Layout for drift augmentation (raw units: m). square obs = object(14: nut_pos 3,
    # nut_quat 4, nut_to_eef_pos 3 [in the gripper frame], nut_to_eef_quat 4) eef_pos(3)
    # eef_quat(4, xyzw) gripper(2); action = OSC_POSE delta, 1.0 = act_scale metres.
    drift_spec = None
    if "square" in str(dataset_path) or "can" in str(dataset_path):
        z = dataset.normalized_train_data["obs"][:, 2]
        z_raw = (z + 1) / 2 * (dataset.stats["obs"]["max"][2] - dataset.stats["obs"]["min"][2]) + dataset.stats["obs"]["min"][2]
        drift_spec = {"obj_pos": [0, 1], "robot_pos": [14, 15], "act_pos": [0, 1],
                      "act_mode": "delta", "act_scale": 0.05,
                      "rel": {"idx": [7, 8, 9], "quat": [17, 18, 19, 20]},
                      "lift": {"z_idx": 2, "z_rest": float(np.percentile(z_raw, 5)), "eps": 0.02},
                      "unit": "m"}
    return (env, dataset, make_dataloader, normalize_data, unnormalize_data,
            initial_action, drift_spec)


def setup(task, dataset_path=None, batch_size=1024, num_workers=1,
          need_loader=True, env_seed=500, perturb_level=0.0,
          need_render=False):
    """Build env + dataset + dataloader and return them in a namespace dict.

    Both task families expose the same contract, so everything downstream of
    this function is task-agnostic:
      * env speaks the Push-T API (reset -> (obs, info), step -> 5-tuple)
      * dataset items carry obs / obs_seq / action
      * g["initial_action"](obs, action_dim) gives the flow's starting action
    """
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}; expected one of {TASKS}")
    dataset_path = dataset_path or DEFAULT_DATASET[task]

    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    builder = _build_pusht if task == "pusht" else _build_robomimic
    (env, dataset, make_dataloader, normalize_data, unnormalize_data,
     initial_action, drift_spec) = builder(dataset_path, pred_horizon, obs_horizon,
                                           action_horizon, env_seed, perturb_level,
                                           need_render)

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
        "drift_spec": drift_spec,
        "pred_horizon": pred_horizon,
        "obs_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "normalize_data": normalize_data,
        "unnormalize_data": unnormalize_data,
        "initial_action": initial_action,
    }


# =============================================================================
# 4. Model
# =============================================================================
def _quat_xyzw_to_rot(q):
    """(..., 4) xyzw -> (..., 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def augment_drift(nobs_seq, naction, spec, stats, obs_horizon, rho_max, p_aug=0.5):
    """Synthesize exogenous object drift in a demo window, relabelling the demo so
    it stays a valid demonstration of the *same* skill in a moving world.

    Every window gets a constant-velocity drift d = rho*u (rho ~ U(0, rho_max) raw
    units/step with prob p_aug, else 0; u uniform on the circle). With j0 the chunk
    start (obs_horizon-1) and C_j the number of un-lifted steps between j0 and j:
        object_pos[j]  += d*C_j            the world moves
        robot_pos[j]   += d*C_{j-1}        the demonstrator followed (one step behind,
                                           as the eval env applies drift after the step)
        action[j]      += d*C_j            (absolute targets)   or
        action[j]      += d*m_j/act_scale  (delta commands: extra per-step motion)
        rel[j]         += R_g(q_j)^T [d*m_j, 0]   object-in-gripper offset (square)
    For a static demo the map is the identity, so it cannot change what the policy
    does at rest; it only adds the regime the demos lack. Operates on normalized
    tensors (B, L, O)/(B, L, A) and returns augmented copies.
    """
    B, L, O = nobs_seq.shape
    dev = nobs_seq.device
    j0 = obs_horizon - 1
    omin = torch.as_tensor(stats["obs"]["min"], device=dev, dtype=nobs_seq.dtype)
    omax = torch.as_tensor(stats["obs"]["max"], device=dev, dtype=nobs_seq.dtype)
    amin = torch.as_tensor(stats["action"]["min"], device=dev, dtype=naction.dtype)
    amax = torch.as_tensor(stats["action"]["max"], device=dev, dtype=naction.dtype)
    ospan = torch.where(omax - omin == 0, torch.ones_like(omax), omax - omin)
    aspan = torch.where(amax - amin == 0, torch.ones_like(amax), amax - amin)
    o_raw = lambda idx: (nobs_seq[..., idx] + 1) / 2 * ospan[idx] + omin[idx]

    # drift velocity per window
    rho = torch.rand(B, device=dev) * rho_max * (torch.rand(B, device=dev) < p_aug).float()
    th = torch.rand(B, device=dev) * 2 * math.pi
    d = torch.stack([rho * torch.cos(th), rho * torch.sin(th)], dim=-1)          # (B, 2)

    # un-lifted mask m_j and cumulative count C_j relative to the chunk start
    if spec.get("lift"):
        z = o_raw([spec["lift"]["z_idx"]])[..., 0]                                  # (B, L)
        m = (z < spec["lift"]["z_rest"] + spec["lift"]["eps"]).float()
    else:
        m = torch.ones(B, L, device=dev)
    S = torch.cumsum(m, dim=1)                                                     # (B, L)
    C = S - S[:, j0:j0 + 1]                                                        # C_{j0} = 0
    C_prev = torch.cat([C[:, :1] - m[:, :1], C[:, :-1]], dim=1)                    # C_{j-1}
    D_obj = d.unsqueeze(1) * C.unsqueeze(-1)                                       # (B, L, 2)
    D_rob = d.unsqueeze(1) * C_prev.unsqueeze(-1)
    D_step = d.unsqueeze(1) * m.unsqueeze(-1)                                      # D_obj - D_rob

    # keep augmented windows inside the data range (abs tasks: arena walls)
    obj_new = o_raw(spec["obj_pos"]) + D_obj
    rob_new = o_raw(spec["robot_pos"]) + D_rob
    lo_o, hi_o = omin[spec["obj_pos"]], omax[spec["obj_pos"]]
    lo_r, hi_r = omin[spec["robot_pos"]], omax[spec["robot_pos"]]
    ok = ((obj_new >= lo_o) & (obj_new <= hi_o) & (rob_new >= lo_r) & (rob_new <= hi_r)).all(-1).all(-1)
    valid = ok.float().view(B, 1, 1)
    D_obj, D_rob, D_step = D_obj * valid, D_rob * valid, D_step * valid

    nobs = nobs_seq.clone(); nact = naction.clone()
    nobs[..., spec["obj_pos"]] += D_obj * 2 / ospan[spec["obj_pos"]]
    nobs[..., spec["robot_pos"]] += D_rob * 2 / ospan[spec["robot_pos"]]
    if spec.get("rel"):
        R = _quat_xyzw_to_rot(o_raw(spec["rel"]["quat"]))                          # (B, L, 3, 3)
        step3 = torch.cat([D_step, torch.zeros_like(D_step[..., :1])], dim=-1)     # (B, L, 3)
        d_rel = torch.einsum("blij,bli->blj", R, step3)                            # R^T v
        nobs[..., spec["rel"]["idx"]] += d_rel * 2 / ospan[spec["rel"]["idx"]]
    if spec["act_mode"] == "abs":
        nact[..., spec["act_pos"]] += D_obj * 2 / aspan[spec["act_pos"]]
    else:
        nact[..., spec["act_pos"]] += (D_step / spec["act_scale"]) * 2 / aspan[spec["act_pos"]]
    return nobs, nact, ok


def make_cond(win_start, win_now, dual_cond):
    """Flatten observation windows into the conditioning vector.

    win_start, win_now: (..., H, O). Returns (..., cond_dim)."""
    if dual_cond == "concat":
        return torch.cat([win_start.flatten(-2), win_now.flatten(-2)], dim=-1)
    if dual_cond == "residual":
        return torch.cat([win_start.flatten(-2), (win_now - win_start).flatten(-2)], dim=-1)
    return win_now.flatten(-2)


def prefix_len(g):
    """Max number of already-executed actions within one chunk (t_idx in 0..T)."""
    return g["pred_horizon"] - g["obs_horizon"]


def build_nets(g, prefix=False, gru_hidden=64, dual_cond="none", arch="unet", lip=5.0, contr_backbone="unet"):
    """prefix=True adds a GRU over the actions already executed in the current
    chunk and appends its final state to the observation conditioning. The
    executed history is a sufficient statistic for which branch the trajectory
    is on, so it gives the field mode memory while h_t supplies the current
    state. A GRU (rather than a flattened, zero-padded window) keeps the
    conditioning size independent of action_dim and chunk length."""
    obs_dim = g["obs"].shape[-1]
    action_dim = g["action"].shape[-1]
    obs_horizon = g["obs_horizon"]
    # dual_cond adds the chunk-start window as a reference next to the current one:
    #   concat   [h0, h_t]          residual   [h0, h_t - h0]
    # Without a reference the network cannot tell world change (innovation) from
    # its own tracking noise, so it reacts to both; with h0 it can learn the gate.
    cond_obs = obs_dim * obs_horizon * (2 if dual_cond != "none" else 1) + (gru_hidden if prefix else 0)


    if arch == "contractive":
        # stabilizing term -k(a - m) hard-wired; residual psi is lip-Lipschitz in a
        velocity_net = ContractiveField(action_dim, cond_obs, k=10.0, lip=lip,
                                        backbone=contr_backbone, unet_cls=ConditionalUnet1D)
    else:
        velocity_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_obs,   # anchor
            updownsample_type="Linear",
            sin_embedding_scale=100,
        )

    mods = {"velocity_net": velocity_net}
    if prefix:
        mods["traj_gru"] = nn.GRU(input_size=action_dim, hidden_size=gru_hidden,
                                  num_layers=1, batch_first=True)
    return nn.ModuleDict(mods)


def encode_prefix(nets, seq, lengths):
    """GRU state after the first `lengths[b]` steps of seq[b]; zeros if 0.

    seq: (B, L, A) zero-padded; lengths: (B,) ints in [0, L]. The GRU is causal,
    so padding after `lengths` cannot affect the gathered state.
    """
    B, L, _ = seq.shape
    out, _ = nets["traj_gru"](seq)                                   # (B, L, H)
    idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, out.shape[-1])
    h = out.gather(1, idx).squeeze(1)                                # (B, H)
    return h * (lengths > 0).unsqueeze(-1).to(h.dtype)


def load_ema_nets(g, task, ckpt, device, ckpt_path=None, tag="", prefix=False, gru_hidden=64, dual_cond="none",
                  arch=None, lip=None):
    """Rebuild the model and load an EMA checkpoint into eval mode.

    ckpt_path, when given, names the file directly. That lets one script
    score a checkpoint trained by the other (the architectures match), e.g.
    an SFP model executed under the CL rollout. The field architecture
    (unet / contractive) is read from the checkpoint meta unless given.
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
    meta = {k: v for k, v in blob.items() if k != "state_dict"} if isinstance(blob, dict) and "state_dict" in blob else {}
    arch = arch or meta.get("arch", "unet") or "unet"
    lip = float(meta.get("lip", 5.0) if lip is None else lip)
    cb = meta.get("contr_backbone", "mlp") if arch == "contractive" else "unet"   # pre-backbone ckpts were mlp
    nets = build_nets(g, prefix=prefix, gru_hidden=gru_hidden, dual_cond=dual_cond, arch=arch, lip=lip, contr_backbone=cb).to(device)
    ema = EMAModel(model=nets, power=0.75)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    ema.averaged_model.load_state_dict(sd)
    return ema.averaged_model.to(device).eval()


# =============================================================================
# 4b. Checkpoint metadata
# =============================================================================
def ckpt_meta(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return {k: v for k, v in blob.items() if k != "state_dict"}
    return {}


# =============================================================================
# 5. Training
# =============================================================================
def train(g, task, epochs, smoke, device=None, train_seed=0, tag="",
          consist_lambda=0.0, prefix=False, prefix_noise=0.05, gru_hidden=64,
          dual_cond="none", sigma_train=None, obs_shift=0, sigma_min=None,
          drift_aug=0.0, drift_p=0.5, arch="unet", lip=5.0, contr_backbone="unet",
          stale_p=0.0):
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

    nets = build_nets(g, prefix=prefix, gru_hidden=gru_hidden, dual_cond=dual_cond, arch=arch, lip=lip, contr_backbone=contr_backbone).to(device)
    print(f"arch={arch}" + (f" (backbone={contr_backbone}, lip={lip}, guaranteed contraction >= {10.0 - lip:g})" if arch == "contractive" else "")
          + f"  params={sum(p.numel() for p in nets.parameters())/1e6:.2f}M")
    ema = EMAModel(model=nets, power=0.75)
    opt = torch.optim.AdamW(nets.parameters(), lr=1e-4, weight_decay=1e-6)
    sched = get_scheduler("cosine", optimizer=opt,
                          num_warmup_steps=len(dataloader) * 10,
                          num_training_steps=len(dataloader) * epochs)
    T = pred_horizon - obs_horizon
    spec = g.get("drift_spec")
    if drift_aug > 0 and spec is None:
        raise ValueError(f"--drift-aug: no drift_spec for task {task!r}")
    if drift_aug > 0:
        print(f"drift augmentation: rho_max={drift_aug} {spec['unit']}/step, p={drift_p}, spec={ {k: v for k, v in spec.items() if k != 'rel'} }")

    for epoch in tqdm(range(epochs), desc=f"CL-SFP-{task}"):
        losses = []
        for nbatch in dataloader:
            nobs_seq = nbatch["obs_seq"].to(device)
            naction = nbatch["action"].to(device)
            if drift_aug > 0:
                nobs_seq, naction, _ = augment_drift(nobs_seq, naction, spec, g["stats"],
                                                     obs_horizon, drift_aug, drift_p)
            xi = naction[:, obs_horizon - 1:, :]
            B = xi.shape[0]
            t = torch.rand(B, device=device)

            xit, dxidt = LinearlyInterpolateTrajectory(xi, t)
            a, v = SampleCFMInputsAndTargets(xit, dxidt, t, k, sigma0, σ_const=sigma_train, σ_min=sigma_min)
            a, v = a.unsqueeze(1), v.unsqueeze(1)

            b_idx = torch.arange(B, device=device).unsqueeze(1)
            # current = time-aligned obs window (same as CL-SFP)
            t_idx = (t * T).long().clamp(0, T)
            ar = torch.arange(obs_horizon, device=device)
            # obs_shift=0: the window whose last obs is the one xi[t_idx] was *chosen* at.
            # obs_shift=1: one step later - the obs *after* xi[t_idx] was executed, which is
            # exactly the window the closed-loop rollout has in hand at flow time t_idx*dt.
            win_idx = (t_idx + obs_shift).clamp(0, T)
            if stale_p > 0:
                # Mixed staleness: with prob stale_p, condition flow time t_idx on an OLDER
                # window h_{t_idx - tau}, tau ~ U{0..t_idx}. The target is still xi at t_idx,
                # so the field must carry plan progression through t (as SFP does with h0)
                # instead of re-reading its position from h at every step. tau = 0 is the
                # CL-SFP pairing, tau = t_idx the SFP pairing: one model spans the family.
                use = torch.rand(B, device=device) < stale_p
                tau = (torch.rand(B, device=device) * (win_idx + 1).float()).long().clamp(max=win_idx)
                win_idx = torch.where(use, win_idx - tau, win_idx)
            cur_idx = win_idx.unsqueeze(1) + ar
            win_now = nobs_seq[b_idx, cur_idx]                                  # (B, H, O)
            win_start = nobs_seq[b_idx, ar.unsqueeze(0).expand(B, -1)]          # (B, H, O) chunk start
            current_flat = make_cond(win_start, win_now, dual_cond)             # (B, cond)
            if prefix:
                # executed actions of this chunk before the current one: xi[0:t_idx].
                # Noise on the demo prefix stands in for the policy's own imperfect
                # history at inference (exposure bias).
                Tn = xi.shape[1] - 1
                pos = torch.arange(Tn, device=device).unsqueeze(0)          # (1, T)
                mask = (pos < t_idx.unsqueeze(1)).unsqueeze(-1)              # (B, T, 1)
                pref = xi[:, :Tn, :] * mask
                if prefix_noise > 0:
                    pref = pref + prefix_noise * torch.randn_like(pref) * mask
                pref_h = encode_prefix(nets, pref, t_idx)                    # (B, H)
                current_flat = torch.cat([current_flat, pref_h], dim=1)

            vhat = nets["velocity_net"](sample=a, timestep=t,
                                        global_cond=current_flat)
            loss = nn.functional.mse_loss(v, vhat)
            if consist_lambda > 0:
                # Penalise sensitivity of the field to a one-step-older window:
                # a direct handle on L_h in the closed-loop deviation bound.
                prev_idx = (t_idx - 1).clamp(0, T).unsqueeze(1) + ar
                prev_flat = make_cond(win_start, nobs_seq[b_idx, prev_idx], dual_cond)
                if prefix:
                    prev_flat = torch.cat([prev_flat, pref_h], dim=1)
                v_prev = nets["velocity_net"](sample=a, timestep=t, global_cond=prev_flat)
                loss = loss + consist_lambda * nn.functional.mse_loss(vhat, v_prev)

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
                        "consist_lambda": consist_lambda,
                        "prefix": prefix, "prefix_noise": prefix_noise,
                        "prefix_encoder": "gru", "gru_hidden": gru_hidden,
                        "dual_cond": dual_cond, "sigma_train": sigma_train,
                        "obs_shift": obs_shift, "sigma_min": sigma_min,
                        "drift_aug": drift_aug, "drift_p": drift_p,
                        "arch": arch, "lip": lip, "contr_backbone": contr_backbone,
                        "stale_p": stale_p,
                        "epoch": epoch + 1, "task": task}, path)
            print(f"[ep{epoch + 1}] loss={np.mean(losses):.4f}", flush=True)
        if smoke:
            print(f"[smoke] epoch {epoch} loss={np.mean(losses):.4f}")
    print(f"train done  (seed={train_seed} tag={tag!r} consist={consist_lambda} prefix={prefix} dual={dual_cond} sigma_train={sigma_train} sigma_min={sigma_min} obs_shift={obs_shift} drift_aug={drift_aug} arch={arch} stale_p={stale_p} -> {ckpt_dir})")


# =============================================================================
# 6. Closed-loop inference
# =============================================================================
def rollout(g, ema_nets, env, seed=0, perturb_level=0.0, max_steps=250,
            action_horizon=8, save_vis=False, device=None, prefix=False, dual_cond="none",
            obs_lag=0, freeze_dims=None, obs_noise=0.0, act_noise=0.0):
    """Run one closed-loop episode. Returns (score, frames, steps).

    obs_noise: std of Gaussian noise added to every conditioning window at inference
               (normalized obs units); a robustness probe for the field's L_h.

    obs_lag=0: the velocity at flow time i*dt is conditioned on the window read
               *after* executing a(i*dt)  (one step fresher than obs_shift=0 training).
    obs_lag=1: conditioned on the window read *before* executing a(i*dt), which is
               the window obs_shift=0 training pairs with t_idx=i.
    """
    device = device or torch.device("cuda")
    obs_horizon = g["obs_horizon"]
    pred_horizon = g["pred_horizon"]
    action_dim = g["action"].shape[-1]
    stats = g["stats"]
    normalize_data = g["normalize_data"]
    unnormalize_data = g["unnormalize_data"]

    obs, info = env.reset(seed_=seed, perturb_level=perturb_level)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render()] if save_vis else []
    rewards, done, step_idx = [], False, 0
    gen = None
    if obs_noise > 0 or act_noise > 0:
        gen = torch.Generator(device=device); gen.manual_seed(10_000 + seed)
    def nwin():
        w = torch.from_numpy(normalize_data(np.stack(obs_deque), stats=stats["obs"])).to(device, torch.float32)
        if obs_noise > 0:
            w = w + obs_noise * torch.randn(w.shape, generator=gen, device=device)
        return w

    a0 = g["initial_action"](obs, action_dim)
    na = torch.from_numpy(normalize_data(a0, stats=stats["action"])).to(
        device, torch.float32)
    na_prev = na.unsqueeze(0).unsqueeze(0)

    dt = 1.0 / (pred_horizon - obs_horizon)

    Tn = prefix_len(g)
    while not done:
        na = na_prev
        executed = []          # normalized actions executed so far in this chunk
        # chunk-start window: the reference for dual conditioning (matches training,
        # where win_start = nobs_seq[:, 0:obs_horizon])
        nstart = nwin()
        with torch.no_grad():
            for i in range(action_horizon):
                if obs_lag:
                    # training-consistent window: read before the step
                    ncur = nwin()
                a = unnormalize_data(na.cpu().numpy().squeeze(axis=(0, 1)),
                                     stats=stats["action"])
                executed.append(na.reshape(-1))
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
                # closed loop: re-read the obs window every executed step
                if not obs_lag:
                    ncur = nwin()
                if freeze_dims:
                    # diagnostic: hold these obs dims at their chunk-start value so the
                    # per-step window refreshes only the remaining channels
                    ncur = ncur.clone(); ncur[..., freeze_dims] = nstart[..., freeze_dims]
                current_flat = make_cond(nstart, ncur, dual_cond).unsqueeze(0)
                if prefix:
                    # training used xi[0:t_idx] with t_idx = i: the actions before
                    # the current one -> executed[:-1] here
                    hist = executed[:-1][:Tn]
                    if hist:
                        seq = torch.stack(hist).unsqueeze(0)                          # (1, n, A)
                        pref_h = encode_prefix(ema_nets, seq, torch.tensor([len(hist)], device=device))
                    else:
                        pref_h = torch.zeros(1, ema_nets["traj_gru"].hidden_size, device=device)
                    current_flat = torch.cat([current_flat, pref_h], dim=1)
                t = torch.tensor(i * dt, device=device, dtype=torch.float32)
                nv = ema_nets["velocity_net"](sample=na, timestep=t,
                                              global_cond=current_flat)
                na = na + nv * dt
                if act_noise > 0:
                    # perturb the flow state (actuator / integration error); the
                    # contraction -k(a - xi) is what should damp this
                    na = na + act_noise * torch.randn(na.shape, generator=gen, device=device)
        na_prev = na.detach()

    return (max(rewards) if rewards else 0.0), imgs, step_idx


def sweep_path(task, seeds, out_json=None, action_horizon=None, tag="", obs_lag=0, obs_noise=0.0, act_noise=0.0):
    """One file per (task, method, seed count) - every checkpoint goes in it.

    A non-default execution chunk length gets its own file: chunk=1 and chunk=8
    are different policies at inference and must not be averaged together.
    """
    if out_json:
        return Path(out_json)
    ahtag = "" if action_horizon in (None, 8) else f"_ah{action_horizon}"
    lagtag = "" if not obs_lag else f"_lag{obs_lag}"
    ontag = ("" if not obs_noise else f"_on{obs_noise:g}") + ("" if not act_noise else f"_an{act_noise:g}")
    return (ROOT / "outputs" / task / f"cl_sfp{tag}" / "eval"
            / f"sweep_s{seeds}{ahtag}{lagtag}{ontag}.json")


def _load_sweep(path, task, seeds, g, tag=""):
    """Open an existing sweep file, or start a fresh payload."""
    if path.is_file():
        try:
            payload = json.load(open(path))
            if payload.get("checkpoints"):
                return payload
        except (json.JSONDecodeError, OSError) as exc:
            print(f"could not reuse {path} ({exc}); starting a new sweep")
    return {
        # `rollout` is this script's execution scheme; `trained_by` is the
        # method whose checkpoint is scored. They differ under --ckpt-path
        # (e.g. an SFP checkpoint executed with the CL-SFP rollout).
        "rollout": "cl_sfp",
        "trained_by": "cl_sfp",
        "method": "cl_sfp",   # kept for older readers; equals trained_by
        "tag": tag,
        "task": task,
        "seeds": seeds,
        "action_horizon": g["action_horizon"],
        "pred_horizon": g["pred_horizon"],
        "obs_horizon": g["obs_horizon"],
        "obs_dim": int(g["obs"].shape[-1]),
        "action_dim": int(g["action"].shape[-1]),
        "checkpoints": {},
    }


def _record_eval_horizon(payload, action_horizon):
    payload["eval_action_horizon"] = int(action_horizon)


def _ingest_per_ckpt_files(payload, task, seeds, action_horizon=8):
    """Fold in results written by the older one-file-per-checkpoint layout.

    A sweep that was interrupted (a reboot, a killed job) left those files
    behind. Reading them back means the sweep resumes instead of re-measuring
    checkpoints that already cost hours.

    Only files measured with the same execution chunk length are taken. The
    older files carry no such field and were all run at the trained default
    of 8, so they are ignored for any other chunk length - otherwise a chunk=1
    sweep would silently inherit chunk=8 numbers and skip every rollout.
    """
    d = ROOT / "outputs" / task / f"{payload['method']}{payload.get('tag','')}" / "eval"
    for f in sorted(d.glob(f"ep*_s{seeds}.json")):
        try:
            b = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if int(b.get("eval_action_horizon", 8)) != int(action_horizon):
            continue
        c = str(b.get("ckpt"))
        if c in ("None", "") or c in payload["checkpoints"]:
            continue
        payload["checkpoints"][c] = {
            "results": b.get("results", {}),
            "timestamp": b.get("timestamp"),
            "argv": b.get("argv"),
            "source": f.name,
        }
        print(f"  resumed ep{c} from {f.name}")


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


def eval_one(g, task, seeds, ckpt, perturbs, smoke, device=None,
             action_horizon=8, ckpt_path=None, tag="", prefix=False, gru_hidden=64, dual_cond="none",
             obs_lag=0, freeze_dims=None, on_result=None, obs_noise=0.0, act_noise=0.0):
    """Score one checkpoint over the perturbation levels. No file writing here;
    `on_result(perturb_key, entry)` is called after each level so the caller can
    checkpoint partial progress.

    Per-seed scores are kept, not just the mean: two runs share the seed list,
    so the individual episodes are what allow a later paired comparison.
    """
    device = device or torch.device("cuda")
    env = g["env"]
    ema_nets = load_ema_nets(g, task, ckpt, device, ckpt_path=ckpt_path, tag=tag, prefix=prefix, gru_hidden=gru_hidden, dual_cond=dual_cond)

    results = {}
    for perturb in perturbs:
        scores, steps = [], []
        for s in tqdm(range(seeds), desc=f"ep{ckpt} perturb{perturb}",
                      leave=False):
            env.seed(s)
            score, _, nsteps = rollout(g, ema_nets, env, seed=s,
                               perturb_level=perturb, device=device,
                               action_horizon=action_horizon, prefix=prefix, dual_cond=dual_cond,
                               obs_lag=obs_lag, freeze_dims=freeze_dims, obs_noise=obs_noise, act_noise=act_noise)
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
             ckpt_path=None, tag="", obs_lag=0, freeze_dims=None, obs_noise=0.0, act_noise=0.0):
    """Evaluate one or more checkpoints into a single sweep file.

    The file is rewritten after every checkpoint, so an interrupted sweep keeps
    what it has already measured and a later run picks up where it stopped.
    """
    if not isinstance(ckpts, (list, tuple)):
        ckpts = [ckpts]

    path = sweep_path(task, seeds, out_json, action_horizon, tag, obs_lag, obs_noise, act_noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_sweep(path, task, seeds, g, tag)
    if ckpt_path:
        payload["checkpoints"] = {}   # a foreign checkpoint: start clean
    # Resume only into the default sweep file. A custom --out-json or a
    # foreign --ckpt-path describes a run nothing in this directory matches,
    # so pulling old files in would silently substitute another policy's
    # numbers and skip the rollouts.
    if not out_json and not ckpt_path and not obs_lag and not obs_noise and not act_noise:   # lag/noise runs are different policies
        _ingest_per_ckpt_files(payload, task, seeds, action_horizon)
    _record_eval_horizon(payload, action_horizon)
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
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2)

        # architecture must match the checkpoint: read the prefix flag from its meta
        probe = Path(str(ckpt_path)) if ckpt_path else (ROOT / "outputs" / task / f"cl_sfp{tag}" / f"ep{ckpt}.ckpt")
        meta = ckpt_meta(str(probe)) if probe.is_file() else {}
        use_prefix = bool(meta.get("prefix", False))
        gh = int(meta.get("gru_hidden", 64))
        dc = meta.get("dual_cond", "none") or "none"
        payload["prefix"] = use_prefix
        payload["dual_cond"] = dc
        payload["obs_lag"] = int(obs_lag)
        payload["obs_noise"] = float(obs_noise)
        payload["act_noise"] = float(act_noise)
        if freeze_dims:
            payload["freeze_dims"] = list(freeze_dims)
        payload["obs_shift"] = int(meta.get("obs_shift", 0) or 0)
        if meta.get("sigma_min") is not None:
            payload["sigma_min"] = meta["sigma_min"]
        if meta.get("drift_aug"):
            payload["drift_aug"] = meta["drift_aug"]; payload["drift_p"] = meta.get("drift_p")
        if meta.get("stale_p"):
            payload["stale_p"] = meta["stale_p"]
        payload["arch"] = meta.get("arch", "unet") or "unet"
        if payload["arch"] == "contractive":
            payload["lip"] = meta.get("lip")
        if meta.get("sigma_train") is not None:
            payload["sigma_train"] = meta["sigma_train"]
        if use_prefix:
            payload["prefix_encoder"] = meta.get("prefix_encoder", "gru"); payload["gru_hidden"] = gh
        results = eval_one(g, task, seeds, ckpt, todo, smoke, device=device,
                           action_horizon=action_horizon, ckpt_path=ckpt_path, tag=tag,
                           prefix=use_prefix, gru_hidden=gh, dual_cond=dc, obs_lag=obs_lag,
                           freeze_dims=freeze_dims, on_result=_save, obs_noise=obs_noise, act_noise=act_noise)
        merged.update(results)
        payload["checkpoints"][key] = {
            "results": merged,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv[1:],
        }
        _stamp(payload, args)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    # --- ranking and best pick -------------------------------------------
    # Also run when every checkpoint was skipped, so the bookkeeping fields
    # exist even on a pure resume.
    _stamp(payload, args)
    ranked = sorted(
        ((int(c), _score_of(e, perturbs)) for c, e in
         payload["checkpoints"].items()),
        key=lambda kv: (-kv[1] if kv[1] == kv[1] else 1, kv[0]))
    ranked = [(c, s) for c, s in ranked if s == s]  # drop NaN

    if ranked:
        payload["ranking"] = [{"ckpt": c, "score": s} for c, s in ranked]
        best_ckpt, best_score = ranked[0]
        payload["best"] = {
            "ckpt": best_ckpt,
            "score": best_score,
            "metric": f"mean over perturbs {[float(p) for p in perturbs]}",
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

        print()
        print(f"  {'ckpt':>6}  {'score':>8}  {'95% CI':>9}")
        print("  " + "-" * 27)
        for c, s in sorted(ranked, key=lambda kv: kv[0]):
            e = payload["checkpoints"][str(c)]["results"]
            ci = np.mean([e[str(p)]["ci95"] for p in perturbs
                          if str(p) in e]) if e else float("nan")
            flag = "  <- best" if c == best_ckpt else ""
            print(f"  {c:>6}  {s:>8.4f}  {ci:>9.4f}{flag}")
        print()
        print(f"BEST ep{best_ckpt}  score={best_score:.4f}  "
              f"({payload['best']['metric']})")

    print(f"saved -> {path}")
    return payload


def visualize(g, task, ckpt, seed, perturb, out, device=None):
    """Run a single episode and save it as a video (mp4, gif fallback)."""
    device = device or torch.device("cuda")
    env = g["env"]
    ema_nets = load_ema_nets(g, task, ckpt, device)

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
    """'1000' -> 1000 | '100-500' -> [100,200,...,500] | '100,300' -> [100,300]"""
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
    ap.add_argument("--ckpt", default="1000")
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
    ap.add_argument("--prefix", action="store_true",
                    help="condition on the executed-action history of the current chunk")
    ap.add_argument("--dual-cond", default="none", choices=["none", "concat", "residual"],
                    help="add the chunk-start window as a reference: concat [h0,h_t] or "
                         "residual [h0, h_t-h0]. Lets the network learn to ignore its own "
                         "tracking noise while still reacting to world change.")
    ap.add_argument("--obs-shift", type=int, default=0,
                    help="train: pair flow time t_idx with the window t_idx+shift (1 = the "
                         "obs after xi[t_idx] executed, matching the default rollout)")
    ap.add_argument("--freeze-dims", default=None,
                    help="diagnostic (eval): comma-separated obs dims held at their chunk-start "
                         "value in the per-step window, e.g. '0,1'. Requires --out-json.")
    ap.add_argument("--act-noise", type=float, default=0.0,
                    help="eval: std of Gaussian noise added to the flow state a after every step "
                         "(normalized action units); probes the contraction directly")
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help="eval: std of Gaussian noise on the normalized conditioning window "
                         "(robustness probe); results go to sweep_*_on<std>.json")
    ap.add_argument("--obs-lag", type=int, default=0,
                    help="eval: 1 = condition v(t=i*dt) on the window read before executing "
                         "a(i*dt) (matches obs_shift=0 training); 0 = after (default)")
    ap.add_argument("--arch", default="unet", choices=["unet", "contractive"],
                    help="contractive: v = mdot - k(a-m) + lip*psi with psi lip-Lipschitz in a, "
                         "so dv/da <= -(k-lip) I at every flow time by construction")
    ap.add_argument("--contr-backbone", default="unet", choices=["unet", "mlp"],
                    help="backbone predicting the plan (m, mdot) in the contractive arch")
    ap.add_argument("--lip", type=float, default=5.0,
                    help="Lipschitz budget of the residual psi (contractive arch); contraction >= k-lip")
    ap.add_argument("--stale-p", type=float, default=0.0,
                    help="train: fraction of samples whose window is an older h_{t-tau}, tau~U{0..t} "
                         "(keeps plan progression learnable; 0 = pure time-aligned CL-SFP)")
    ap.add_argument("--drift-aug", type=float, default=0.0,
                    help="train: synthesize constant-velocity object drift up to this many raw "
                         "units/step (px for pusht, m for robomimic) with equivariant relabelling")
    ap.add_argument("--drift-p", type=float, default=0.5,
                    help="fraction of training windows that receive drift")
    ap.add_argument("--sigma-min", type=float, default=None,
                    help="floor on the training spread: sigma_t = max(sigma0*exp(-k t), sigma_min) "
                         "(normalized units). Keeps late-t contraction learnable.")
    ap.add_argument("--sigma-train", type=float, default=None,
                    help="constant training spread of a around xi(t) (normalized units), "
                         "replacing the collapsing sigma0*exp(-k t)")
    ap.add_argument("--gru-hidden", type=int, default=64,
                    help="hidden size of the prefix GRU")
    ap.add_argument("--prefix-noise", type=float, default=0.05,
                    help="train-time Gaussian noise on the demo prefix (exposure bias)")
    ap.add_argument("--consist-lambda", type=float, default=0.0,
                    help="weight of the one-step conditioning-consistency penalty")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="training RNG seed (init + data order)")
    ap.add_argument("--tag", default="",
                    help="suffix on the checkpoint dir, e.g. _seed1 -> outputs/<task>/cl_sfp_seed1/")
    ap.add_argument("--ckpt-path", default=None,
                    help="load this checkpoint file instead of outputs/<task>/<method>/. "
                         "Requires --out-json so the result does not land in a sweep file.")
    ap.add_argument("--action-horizon", type=int, default=8,
                    help="actions executed per chunk at eval time; the model "
                         "is trained for 8. 1 = replan every step.")
    ap.add_argument("--redo", action="store_true",
                    help="re-measure checkpoints already in the sweep file")
    ap.add_argument("--out-json", default=None,
                    help="where to write eval results "
                         "(default outputs/<task>/cl_sfp/eval/sweep_s<seeds>.json)")
    args = ap.parse_args()
    if args.ckpt_path and not args.out_json:
        ap.error("--ckpt-path needs --out-json")
    freeze_dims = [int(x) for x in args.freeze_dims.split(",")] if args.freeze_dims else None
    if freeze_dims and not args.out_json:
        ap.error("--freeze-dims needs --out-json")

    device = torch.device(args.device)
    g = setup(args.task,
              dataset_path=args.dataset,
              batch_size=args.batch_size,
              num_workers=args.num_workers,
              need_loader=(args.mode == "train"),
              need_render=(args.mode == "video"))
    print("shared infra loaded; obs_dim=%d action_dim=%d"
          % (g["obs"].shape[-1], g["action"].shape[-1]))

    if args.mode == "train":
        train(g, args.task, args.epochs, args.smoke, device=device,
              train_seed=args.train_seed, tag=args.tag,
              consist_lambda=args.consist_lambda,
              prefix=args.prefix, prefix_noise=args.prefix_noise,
              gru_hidden=args.gru_hidden,
              dual_cond=args.dual_cond, sigma_train=args.sigma_train,
              obs_shift=args.obs_shift, sigma_min=args.sigma_min,
              drift_aug=args.drift_aug, drift_p=args.drift_p,
              arch=args.arch, lip=args.lip, contr_backbone=args.contr_backbone,
              stale_p=args.stale_p)
    elif args.mode == "video":
        ckpt = parse_ckpt(args.ckpt)
        if isinstance(ckpt, list):
            ckpt = ckpt[-1]
        out = args.out or (f"{args.task}_cl_sfp_ep{ckpt}"
                           f"_seed{args.seed}_p{args.perturb}.mp4")
        visualize(g, args.task, ckpt, args.seed, args.perturb, out,
                  device=device)
    else:
        ckpt = parse_ckpt(args.ckpt)
        ckpts = ckpt if isinstance(ckpt, list) else [ckpt]
        evaluate(g, args.task, args.seeds, ckpts, parse_perturb(args.perturbs),
                 args.smoke, device=device, out_json=args.out_json,
                 args=args, redo=args.redo, action_horizon=args.action_horizon,
                 ckpt_path=args.ckpt_path, tag=args.tag, obs_lag=args.obs_lag,
                 freeze_dims=freeze_dims, obs_noise=args.obs_noise, act_noise=args.act_noise)


if __name__ == "__main__":
    main()