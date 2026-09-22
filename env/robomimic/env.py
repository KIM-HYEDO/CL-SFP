"""
env.py - robomimic / robosuite low-dim environment, wrapped to the Push-T API.

env/pusht/pusht_env.py exposes

    env.seed(s)
    obs, info = env.reset(seed_=..., perturb_level=...)
    obs, reward, terminated, truncated, info = env.step(action)
    img = env.render()

Native robomimic returns `obs` from reset and a 4-tuple from step, so the
shared rollout in exp/ would need a per-task branch. This wrapper matches the
Push-T signatures instead, keeping the algorithm code single-path.

Perturbation
------------
`perturb_level` mirrors Push-T, where the T block is displaced every step: here
the manipulated object's free joint is translated by `perturb_level` metres per
step (a steady drift), so the world the policy is looking at keeps changing.
The drift direction is a unit vector in the table plane drawn once per episode
from the seeded RNG, so it is fixed within an episode, differs across seeds,
and is identical for every method evaluated on the same seed. (Push-T drifts
along the fixed +x,+y diagonal instead.) The drift is applied only while the
object is still on the table (z within 2 cm of its reset height); teleporting
an object out of a closed gripper would be unphysical and would test nothing
about reactivity.

`perturb_kind="action"` keeps the older alternative - zero-mean Gaussian noise
on the commanded action - which is an actuator disturbance, not a world change.
The two are not interchangeable and must not share an axis in a plot.
"""

import os
from typing import List, Optional

import numpy as np

try:
    import gym
    from gym.spaces import Box
except ImportError:  # pragma: no cover - gym is a hard dependency in practice
    raise

# Low-dim keys, concatenated in this order. The per-task lists live in
# env/robomimic/tasks.py so that this file and data.py cannot drift apart;
# this name stays as the single-arm default for callers that pass no task.
from env.robomimic.tasks import (OBS_KEYS, PERTURB_OBJECT,
                                 obs_keys as obs_keys_for_task)  # noqa: E402

DEFAULT_OBS_KEYS: List[str] = list(OBS_KEYS["can"])


class RobomimicLowdimWrapper(gym.Env):
    """Flatten robomimic's dict observation and speak the Push-T API."""

    def __init__(self,
                 env,
                 obs_keys: Optional[List[str]] = None,
                 init_state: Optional[np.ndarray] = None,
                 render_hw=(256, 256),
                 render_camera_name="agentview",
                 perturb_level: float = 0.0,
                 perturb_kind: str = "object",
                 perturb_object=None):
        self.env = env
        self.obs_keys = list(obs_keys or DEFAULT_OBS_KEYS)
        # Name (or names, tried in order) of the free joint the drift moves.
        # None means "the one object in the workspace"; see _resolve_object_joint.
        self.perturb_object = perturb_object
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.perturb_level = float(perturb_level)
        if perturb_kind not in ("object", "action"):
            raise ValueError(f"perturb_kind must be 'object' or 'action', got {perturb_kind!r}")
        self.perturb_kind = perturb_kind
        self._obj_joint = None      # resolved at reset: the task object's free joint
        self._obj_z0 = None         # its height right after reset
        self._drift_dir = np.zeros(2)  # per-episode unit drift direction (x, y)
        self.lift_eps = 0.02        # metres above z0 that counts as "lifted"
        self._seed = None
        self._rng = np.random.default_rng(0)

        low = np.full(env.action_dimension, fill_value=-1.0)
        high = np.full(env.action_dimension, fill_value=1.0)
        self.action_space = Box(low=low, high=high, shape=low.shape,
                                dtype=np.float64)

        obs_example = self.get_observation()
        self.observation_space = Box(
            low=np.full_like(obs_example, -1.0),
            high=np.full_like(obs_example, 1.0),
            shape=obs_example.shape,
            dtype=obs_example.dtype,
        )

    # -- observation -------------------------------------------------------
    def get_observation(self) -> np.ndarray:
        raw_obs = self.env.get_observation()
        return np.concatenate([raw_obs[key] for key in self.obs_keys],
                              axis=0).astype(np.float32)

    # -- object perturbation ----------------------------------------------
    @property
    def sim(self):
        return self.env.env.sim          # robomimic EnvRobosuite -> robosuite env -> MjSim

    def _in_workspace_free_joints(self):
        """Free (7-dof) joints whose body sits in the workspace.

        robosuite parks the objects a task does not use far away, e.g. at
        (10, 10, 10), so this is what separates the real objects from the props.
        """
        sim = self.sim
        found = []
        for jid in range(sim.model.njnt):
            if sim.model.jnt_type[jid] != 0:          # mjJNT_FREE == 0
                continue
            name = sim.model.joint_id2name(jid)
            if np.all(np.abs(sim.data.get_joint_qpos(name)[:3]) < 3.0):
                found.append(name)
        return found

    def _resolve_object_joint(self):
        """Pick the free joint the drift displaces.

        can/lift/square leave exactly one object in the workspace, so there is
        nothing to choose and `perturb_object` stays None. transport and
        tool_hang leave several, and the drift is this experiment's independent
        variable - perturbing the lid rather than the payload would silently be
        a different experiment under the same name. So a multi-object task must
        name its target, and an unnamed ambiguity is an error, never a guess.
        """
        found = self._in_workspace_free_joints()
        if not found:
            raise RuntimeError("no in-workspace free joint found for object perturbation")

        if self.perturb_object is None:
            if len(found) > 1:
                raise RuntimeError(
                    f"{len(found)} in-workspace free joints ({', '.join(found)}); "
                    f"the drift target is ambiguous. Name it in "
                    f"env/robomimic/tasks.py PERTURB_OBJECT.")
            best = found[0]
        else:
            patterns = ([self.perturb_object] if isinstance(self.perturb_object, str)
                        else list(self.perturb_object))
            best = next((n for p in patterns for n in found if p in n), None)
            if best is None:
                raise RuntimeError(
                    f"no in-workspace free joint matches {patterns}; "
                    f"available: {', '.join(found)}")

        self._obj_joint = best
        self._obj_z0 = float(self.sim.data.get_joint_qpos(best)[2])

    def _perturb_object(self):
        q = self.sim.data.get_joint_qpos(self._obj_joint).copy()
        if q[2] - self._obj_z0 > self.lift_eps:       # grasped and lifted: leave it
            return
        q[:2] += self.perturb_level * self._drift_dir
        self.sim.data.set_joint_qpos(self._obj_joint, q)
        self.sim.forward()

    # -- Push-T compatible API --------------------------------------------
    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
        self._rng = np.random.default_rng(0 if seed is None else seed)

    def reset(self, seed_=None, perturb_level=None, options=None):
        """Returns (obs, info), like PushTEnv.reset."""
        if seed_ is not None:
            self.seed(seed_)
        if perturb_level is not None:
            self.perturb_level = float(perturb_level)

        if self.init_state is not None:
            self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            np.random.seed(seed=self._seed)
            self.env.reset()
            self._seed = None
        else:
            self.env.reset()

        if self.perturb_kind == "object":
            self._resolve_object_joint()
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            self._drift_dir = np.array([np.cos(theta), np.sin(theta)])
        return self.get_observation(), {}

    def step(self, action):
        """Returns (obs, reward, terminated, truncated, info), like PushTEnv."""
        action = np.asarray(action, dtype=np.float64)
        if self.perturb_level > 0 and self.perturb_kind == "action":
            action = action + self._rng.normal(
                scale=self.perturb_level, size=action.shape)
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

        _, reward, done, info = self.env.step(action)
        if self.perturb_level > 0 and self.perturb_kind == "object":
            self._perturb_object()                    # world changes after the step, like Push-T
        return self.get_observation(), reward, bool(done), False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(mode=mode, height=h, width=w,
                               camera_name=self.render_camera_name)


def make_env(dataset_path: str,
             obs_keys: Optional[List[str]] = None,
             perturb_level: float = 0.0,
             render_offscreen: bool = True,
             perturb_kind: str = "object",
             task: Optional[str] = None) -> RobomimicLowdimWrapper:
    """Build the simulator described by the demonstration file's metadata.

    The environment definition (task, robot, controller) is read out of the
    HDF5 itself, so the sim always matches the demonstrations it is scored
    against. `task` supplies what the HDF5 cannot: which observation keys this
    task concatenates and which object the drift displaces (env/robomimic/tasks.py).
    """
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"robomimic dataset not found: {dataset_path}\n"
            f"Expected env/robomimic/data/<task>/low_dim.hdf5")

    if obs_keys is None:
        obs_keys = obs_keys_for_task(task) if task else list(DEFAULT_OBS_KEYS)
    obs_keys = list(obs_keys)
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {"low_dim": obs_keys, "rgb": []})

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=False,
    )
    return RobomimicLowdimWrapper(env=robomimic_env, obs_keys=obs_keys,
                                 perturb_level=perturb_level, perturb_kind=perturb_kind,
                                 perturb_object=PERTURB_OBJECT.get(task))
