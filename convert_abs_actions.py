"""Add absolute end-effector actions (`actions_abs`) to a robomimic low-dim file.

Same procedure as robomimic's robosuite_add_absolute_actions.py (itself taken
from Diffusion Policy): for every step, reset the simulator to the recorded
state, hand the recorded delta action to the OSC controller, and read back the
goal pose it computed. That goal, as [pos(3), axis-angle(3), gripper(1)] per
arm, is the absolute action that reproduces the same motion when the controller
runs with control_delta=False. robomimic's own script looks for robosuite-1.5
geom names and fails on these v0.1 files, so this uses our make_env instead.

Writes into the file in place - run it on a COPY (low_dim_abs.hdf5), never on
low_dim.hdf5.

Usage: python convert_abs_actions.py <task> [n_workers]
"""

import multiprocessing as mp
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TASK = sys.argv[1]
PATH = ROOT / f"env/robomimic/data/{TASK}/low_dim_abs.hdf5"
_env = None


def _get_env():
    global _env
    if _env is None:
        from env.robomimic.env import make_env
        from env.robomimic.tasks import obs_keys
        _env = make_env(str(PATH), obs_keys=obs_keys(TASK), render_offscreen=False, task=TASK)
    return _env


def convert_demo(demo):
    env = _get_env()
    with h5py.File(PATH, "r") as f:
        g = f["data"][demo]
        states, actions = g["states"][:], g["actions"][:]
    robots = env.env.env.robots
    per_arm = actions.shape[1] // len(robots)
    out = np.zeros_like(actions)
    for t in range(len(states)):
        env.env.reset_to({"states": states[t]})
        for k, robot in enumerate(robots):
            a = actions[t, k * per_arm:(k + 1) * per_arm]
            robot.control(a, policy_step=True)          # runs the goal generator only
            c = robot.controller
            out[t, k * per_arm:k * per_arm + 3] = c.goal_pos
            out[t, k * per_arm + 3:k * per_arm + 6] = Rotation.from_matrix(c.goal_ori).as_rotvec()
            out[t, k * per_arm + 6:(k + 1) * per_arm] = a[6:]   # gripper untouched
    return demo, out


def main():
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    assert PATH.name == "low_dim_abs.hdf5", "refusing to modify the original file"
    with h5py.File(PATH, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if all("actions_abs" in f["data"][d] for d in demos):
            print("already converted"); return
    with mp.get_context("spawn").Pool(n_workers) as pool:
        results = {}
        for i, (demo, abs_a) in enumerate(pool.imap_unordered(convert_demo, demos), 1):
            results[demo] = abs_a
            if i % 20 == 0 or i == len(demos):
                print(f"{i}/{len(demos)} demos", flush=True)
    with h5py.File(PATH, "a") as f:
        for demo, abs_a in results.items():
            g = f["data"][demo]
            if "actions_abs" in g:
                del g["actions_abs"]
            g.create_dataset("actions_abs", data=abs_a)
    print(f"saved actions_abs -> {PATH}")


if __name__ == "__main__":
    main()
