"""
pusht_data.py - Push-T demonstration dataset, sequence sampling and
normalization helpers.

Extracted from `pusht_cl_sfp.ipynb` so that `exp.py` only holds the learning
code. Import with:

    from pusht_data import PushTDataset, normalize_data, unnormalize_data
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import zarr

__all__ = [
    "SequencePointer",
    "create_sequence_pointers",
    "extract_sequence",
    "get_data_stats",
    "normalize_data",
    "unnormalize_data",
    "PushTDataset",
    "make_dataloader",
]


@dataclass
class SequencePointer:
    """Container for sample extraction pointers."""
    buffer_start_idx: int   # start index in the original data buffer
    buffer_end_idx: int     # end index (exclusive) in the original data buffer
    sample_start_idx: int   # start index within the padded sample
    sample_end_idx: int     # end index within the padded sample

    def __iter__(self):
        return iter([self.buffer_start_idx, self.buffer_end_idx,
                     self.sample_start_idx, self.sample_end_idx])


def create_sequence_pointers(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0,
) -> List[SequencePointer]:
    """
    Create sample indices for extracting fixed-length sequences from a dataset
    made of multiple concatenated episodes, padding at episode boundaries so
    that every timestep can serve as a sequence start.
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(SequencePointer(
                buffer_start_idx,
                buffer_end_idx,
                sample_start_idx,
                sample_end_idx,
            ))
    return indices


def extract_sequence(train_data, sequence_length, ptr: SequencePointer):
    """
    Extract a sequence from the buffer and repeat boundary values as padding so
    the output always has exactly `sequence_length` timesteps.
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[ptr.buffer_start_idx:ptr.buffer_end_idx]
        data = sample
        if (ptr.sample_start_idx > 0) or (ptr.sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if ptr.sample_start_idx > 0:
                data[:ptr.sample_start_idx] = sample[0]
            if ptr.sample_end_idx < sequence_length:
                data[ptr.sample_end_idx:] = sample[-1]
            data[ptr.sample_start_idx:ptr.sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    return {'min': np.min(data, axis=0), 'max': np.max(data, axis=0)}


def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])  # to [0, 1]
    ndata = ndata * 2 - 1  # to [-1, 1]
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2  # to [0, 1]
    data = ndata * (stats['max'] - stats['min']) + stats['min']  # to original
    return data


class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        # Read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # All demonstration episodes are concatenated in the first dimension N
        train_data = {
            'action': dataset_root['data']['action'][:],  # (N, action_dim)
            'obs': dataset_root['data']['state'][:],      # (N, obs_dim)
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.sequence_pointers = create_sequence_pointers(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # Compute statistics and normalize data to [-1, 1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        """Count of all possible segments of the dataset"""
        return len(self.sequence_pointers)

    def __getitem__(self, idx):
        ptr = self.sequence_pointers[idx]
        nsample = extract_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            ptr=ptr,
        )
        nsample['obs_seq'] = nsample['obs']                    # (16, O)
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]  # (2, O)
        return nsample


def make_dataloader(dataset, batch_size=1024, num_workers=1, shuffle=True):
    """Standard training dataloader for `PushTDataset`."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,                        # accelerate cpu-gpu transfer
        persistent_workers=(num_workers > 0),   # don't kill workers each epoch
    )