"""
data.py - robomimic low-dim demonstration dataset for Push-T-style training.

Mirrors env/pusht/pusht_data.py: same sequence-pointer windowing, the same
[-1, 1] min/max normalization, and the same `obs` / `obs_seq` / `action` keys,
so exp/sfp.py and exp/cl_sfp.py can consume either task family unchanged.

Two things differ from the pusht loader:

  * episodes come from a robomimic HDF5 file rather than a zarr replay buffer,
    and the low-dim observation is the concatenation of `obs_keys`;
  * the recorded gripper command is binary (open/close), which puts a step
    discontinuity in the middle of an action chunk. Flow matching integrates a
    velocity field, so a step is not representable. `interpolate_binary_gripper_transitions`
    ramps each switch over a few steps to make it integrable.
"""

from dataclasses import dataclass
from typing import Dict, List

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

# Low-dim keys, concatenated in this order to form the observation, e.g. on can:
#   object 14 | robot0_eef_pos 3 | robot0_eef_quat 4 | robot0_gripper_qpos 2
# The per-task lists live in env/robomimic/tasks.py and are shared with env.py:
# the dataset and the simulator must build the vector the same way, and keeping
# two copies of the order in step is exactly what that file exists to prevent.
from env.robomimic.tasks import OBS_KEYS  # noqa: E402

DEFAULT_OBS_KEYS: List[str] = list(OBS_KEYS["can"])


# =============================================================================
# Sequence windowing (identical to env/pusht/pusht_data.py)
# =============================================================================
@dataclass
class SequencePointer:
    buffer_start_idx: int
    buffer_end_idx: int
    sample_start_idx: int
    sample_end_idx: int

    def __iter__(self):
        return iter([self.buffer_start_idx, self.buffer_end_idx,
                     self.sample_start_idx, self.sample_end_idx])


def create_sequence_pointers(episode_ends: np.ndarray, sequence_length: int,
                             pad_before: int = 0,
                             pad_after: int = 0) -> List[SequencePointer]:
    indices = []
    for i in range(len(episode_ends)):
        start_idx = episode_ends[i - 1] if i > 0 else 0
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            indices.append(SequencePointer(
                buffer_start_idx,
                buffer_end_idx,
                0 + start_offset,
                sequence_length - end_offset,
            ))
    return indices


def extract_sequence(train_data, sequence_length, ptr: SequencePointer):
    result = {}
    for key, input_arr in train_data.items():
        sample = input_arr[ptr.buffer_start_idx:ptr.buffer_end_idx]
        data = sample
        if (ptr.sample_start_idx > 0) or (ptr.sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:],
                            dtype=input_arr.dtype)
            if ptr.sample_start_idx > 0:
                data[:ptr.sample_start_idx] = sample[0]
            if ptr.sample_end_idx < sequence_length:
                data[ptr.sample_end_idx:] = sample[-1]
            data[ptr.sample_start_idx:ptr.sample_end_idx] = sample
        result[key] = data
    return result


# =============================================================================
# Normalization
# =============================================================================
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    return {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}


def normalize_data(data, stats):
    span = stats["max"] - stats["min"]
    # Constant dimensions (min == max) would divide by zero; leave them at 0.
    span = np.where(span == 0, 1.0, span)
    ndata = (data - stats["min"]) / span
    return ndata * 2 - 1


def unnormalize_data(ndata, stats):
    span = stats["max"] - stats["min"]
    span = np.where(span == 0, 1.0, span)
    return (ndata + 1) / 2 * span + stats["min"]


# =============================================================================
# Gripper ramp
# =============================================================================
def interpolate_binary_gripper_transitions(actions: np.ndarray,
                                           episode_ends: np.ndarray,
                                           transition_steps: int = 5,
                                           gripper_dims=(-1,)) -> np.ndarray:
    """Linearly ramp each binary open/close switch over `transition_steps`.

    With transition_steps=5 a 1 -> -1 switch becomes
        1.0, 0.6, 0.2, -0.2, -0.6, -1.0
    A step change has unbounded velocity, so a flow-matching policy cannot
    represent it; the ramp gives the velocity field something finite to follow.

    A two-arm task has one such dimension per arm, and every one of them needs
    the ramp, so this takes a sequence (see tasks.GRIPPER_DIMS).
    """
    actions = actions.copy()
    for gripper_dim in gripper_dims:
        actions = _ramp_one_gripper(actions, episode_ends, transition_steps,
                                    gripper_dim)
    return actions


def _ramp_one_gripper(actions, episode_ends, transition_steps, gripper_dim):
    original_gripper = actions[:, gripper_dim].copy()

    start_idx = 0
    for end_idx in episode_ends:
        g = original_gripper[start_idx:end_idx]
        # Treat anything >= 0 as "open" even if it is not exactly +-1.
        g_bin = np.where(g >= 0, 1.0, -1.0)
        transition_indices = np.where(g_bin[1:] != g_bin[:-1])[0]

        for local_idx in transition_indices:
            before, after = g_bin[local_idx], g_bin[local_idx + 1]
            ramp_start = local_idx
            ramp_end = min(local_idx + transition_steps, len(g_bin) - 1)
            ramp = np.linspace(before, after, ramp_end - ramp_start + 1,
                               dtype=np.float32)
            actions[start_idx + ramp_start:start_idx + ramp_end + 1,
                    gripper_dim] = ramp

        start_idx = end_idx

    return actions


# =============================================================================
# HDF5 -> flat arrays
# =============================================================================
def load_episodes(dataset_path: str,
                  obs_keys: List[str],
                  action_key: str = "actions") -> Dict[str, np.ndarray]:
    """Concatenate every demo into flat (N, D) arrays plus episode boundaries.

    This replaces diffusion_policy's ReplayBuffer, which the original notebook
    reached for through a hard-coded sys.path entry. Only the concatenate-and-
    record-boundaries behaviour was ever used, so it lives here instead.
    """
    obs_chunks, action_chunks, episode_ends = [], [], []
    total = 0

    with h5py.File(dataset_path, "r") as file:
        demos = file["data"]
        # demo_0, demo_1, ... in numeric order, not the HDF5 string order.
        names = sorted(demos.keys(), key=lambda s: int(s.split("_")[1]))
        for name in tqdm(names, desc=f"loading {dataset_path}"):
            demo = demos[name]
            obs = np.concatenate(
                [np.asarray(demo["obs"][key]) for key in obs_keys],
                axis=-1).astype(np.float32)
            actions = np.asarray(demo[action_key]).astype(np.float32)
            assert len(obs) == len(actions), (
                f"{name}: {len(obs)} obs vs {len(actions)} actions")
            obs_chunks.append(obs)
            action_chunks.append(actions)
            total += len(actions)
            episode_ends.append(total)

    return {
        "obs": np.concatenate(obs_chunks, axis=0),
        "action": np.concatenate(action_chunks, axis=0),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
    }


# =============================================================================
# Dataset
# =============================================================================
class RobomimicDataset(torch.utils.data.Dataset):
    """Windowed robomimic low-dim dataset.

    Each item is a dict of normalized arrays:
        obs      (obs_horizon,  obs_dim)     window at the chunk start
        obs_seq  (pred_horizon, obs_dim)     full window, for time-aligned conditioning
        action   (pred_horizon, action_dim)
    """

    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon,
                 obs_keys: List[str] = None, gripper_transition_steps: int = 5,
                 gripper_dims=(-1,), abs_action: bool = False):
        obs_keys = list(obs_keys or DEFAULT_OBS_KEYS)

        # Absolute actions come from `actions_abs` (see convert_abs_actions.py)
        # as [pos, axis-angle, gripper] per arm and are re-expressed with the
        # 6D rotation the policy trains on: 7 -> 10 dims per arm. The gripper
        # stays last, so the ramp below still finds it at -1 (single arm) or at
        # the per-arm offsets in tasks.GRIPPER_DIMS, which are given in the
        # 10-dim layout when abs_action is on.
        raw = load_episodes(dataset_path, obs_keys,
                            action_key="actions_abs" if abs_action else "actions")
        if abs_action:
            from env.robomimic.rotation import abs7_to_abs10
            raw["action"] = abs7_to_abs10(raw["action"]).astype(np.float32)
        self.abs_action = abs_action
        episode_ends = raw["episode_ends"]

        action_data = interpolate_binary_gripper_transitions(
            actions=raw["action"],
            episode_ends=episode_ends,
            transition_steps=gripper_transition_steps,
            gripper_dims=gripper_dims,
        )
        train_data = {"action": action_data, "obs": raw["obs"]}

        self.sequence_pointers = create_sequence_pointers(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        stats, normalized = {}, {}
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized[key] = normalize_data(data, stats[key])

        self.stats = stats
        self.normalized_train_data = normalized
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.obs_keys = obs_keys
        self.episode_ends = episode_ends

        print(f"robomimic dataset: {len(episode_ends)} episodes, "
              f"{len(action_data)} steps, obs_dim={raw['obs'].shape[-1]}, "
              f"action_dim={action_data.shape[-1]}, "
              f"{len(self.sequence_pointers)} windows")

    def __len__(self):
        return len(self.sequence_pointers)

    def __getitem__(self, idx):
        ptr = self.sequence_pointers[idx]
        nsample = extract_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            ptr=ptr,
        )
        nsample["obs_seq"] = nsample["obs"][:, :]
        nsample["obs"] = nsample["obs"][:self.obs_horizon, :]
        return nsample


def make_dataloader(dataset, batch_size=1024, num_workers=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
