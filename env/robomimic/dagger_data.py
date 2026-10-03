"""Dataset of DP-labelled states visited by a rollout policy (see dagger.py)."""
from pathlib import Path

import numpy as np
import torch


class DaggerDataset(torch.utils.data.Dataset):
    """Items identical in layout to RobomimicDataset's."""

    def __init__(self, dir, pred_horizon=16, obs_horizon=2):
        d = np.load(Path(dir) / "labels.npz")
        self.plans, self.items = d["plans"], d["items"]
        offs = np.concatenate([[0], np.cumsum(d["ep_lens"])])
        self.obs, self.offs = d["obs"], offs
        self.pred_h, self.obs_h = pred_horizon, obs_horizon

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, j = self.items[i]
        s = self.offs[e] + j - 1
        seq = self.obs[s:s + self.pred_h]
        return {"obs_seq": seq, "obs": seq[:self.obs_h], "action": self.plans[i]}
