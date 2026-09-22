"""Does the simulator agree with the dataset? Run this before trusting a new task.

Three checks, each answering a different "is the low number a bug?" question:

  demo lengths      - how close the human demonstrations run to the episode
                      budget (tasks.MAX_STEPS). A policy slower than the
                      demonstrator gets cut off before it can succeed.
  normalisation     - any observation/action dimension that is constant in the
                      data divides by zero in min-max normalisation and poisons
                      the whole network with NaN.
  one-step physics  - reset the sim to the recorded state at step t, apply the
                      recorded action, compare with the recorded observation at
                      t+1. This is the version-mismatch test: it measures the
                      simulator's error WITHOUT compounding. Open-loop replay of
                      a whole demo is also reported but is NOT a bug indicator
                      on long precise tasks - sub-millimetre per-step error
                      compounds over 500 steps into more than an insertion
                      tolerance, so replay fails while a closed-loop policy is
                      fine. can replays at 100%, tool_hang at 0%, and the
                      one-step error of the two is identical.

Usage: python check_dataset_env.py <task> [n_replay_demos]
"""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from env.robomimic.data import get_data_stats, load_episodes          # noqa: E402
from env.robomimic.env import make_env                                 # noqa: E402
from env.robomimic.tasks import max_steps, obs_keys                    # noqa: E402


def main():
    task = sys.argv[1]
    n_replay = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    path = ROOT / f"env/robomimic/data/{task}/low_dim.hdf5"
    keys = obs_keys(task)
    rng = np.random.default_rng(0)

    with h5py.File(path) as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        lengths = np.array([f["data"][d]["actions"].shape[0] for d in demos])
        budget = max_steps(task)
        print(f"[{task}] demos={len(lengths)}  length min/med/max = "
              f"{lengths.min()}/{int(np.median(lengths))}/{lengths.max()}   "
              f"> budget({budget}): {np.mean(lengths > budget):.0%}   "
              f"> 0.8*budget: {np.mean(lengths > 0.8 * budget):.0%}")
        succ = [float(f["data"][d]["rewards"][:].max() > 0) for d in demos]
        print(f"[{task}] recorded demos with reward>0: {np.mean(succ):.0%}")

    raw = load_episodes(str(path), keys)
    for name in ("obs", "action"):
        st = get_data_stats(raw[name])
        span = np.asarray(st["max"]) - np.asarray(st["min"])
        const = np.where(span == 0)[0].tolist()
        print(f"[{task}] {name}: dim={span.shape[0]}  constant dims: {const or 'none'}")

    env = make_env(str(path), obs_keys=keys, render_offscreen=False, task=task)

    # where robot0_eef_pos sits in the flat observation, for a metric in metres
    off, pos_slice = 0, None
    with h5py.File(path) as f:
        for k in keys:
            w = f["data"][demos[0]]["obs"][k].shape[1]
            if k == "robot0_eef_pos":
                pos_slice = slice(off, off + w)
            off += w

        err_pos = []
        for d in demos[:8]:
            g = f["data"][d]
            S, A = g["states"][:], g["actions"][:]
            O = np.concatenate([g["obs"][k][:] for k in keys], axis=1)
            for t in rng.choice(len(A) - 1, size=25, replace=False):
                env.init_state = S[t]
                env.reset()
                o1, *_ = env.step(A[t])
                err_pos.append(np.abs(o1 - O[t + 1])[pos_slice].max())
        print(f"[{task}] one-step |sim - recorded| eef_pos (m): "
              f"med {np.median(err_pos):.1e}  p90 {np.percentile(err_pos, 90):.1e}  "
              f"max {np.max(err_pos):.1e}   (can: med ~4e-4)")

        replay = []
        for d in demos[:n_replay]:
            g = f["data"][d]
            env.init_state = g["states"][0]
            env.reset()
            best = 0.0
            for a in g["actions"][:]:
                _, r, *_ = env.step(a)
                best = max(best, r)
            replay.append(float(best > 0))
        print(f"[{task}] open-loop replay of {n_replay} demos: {np.mean(replay):.0%} success "
              f"(informational; see docstring)")
    env.init_state = None


if __name__ == "__main__":
    main()
