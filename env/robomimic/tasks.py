"""
tasks.py - the per-task facts that stopped being uniform once the suite grew.

can/lift/square are single-arm and have exactly one manipulated object, so the
original code could hard-code one observation layout and find the perturbation
target by taking the first free joint it came across. transport is two-arm and
both transport and tool_hang carry several free objects, so neither choice
survives contact with them: the second arm would silently drop out of the
observation, and the drift would land on whichever object MuJoCo happened to
order first. Both are now explicit, and both fail loudly rather than quietly.

Everything here is keyed by the project's task name (the --task argument), not
by the robosuite env name, so algo/ and env/ agree on one vocabulary.
"""

from typing import List

ROBOMIMIC_TASKS = ("can", "lift", "square", "transport", "tool_hang")


# -- observation layout -------------------------------------------------------
# Concatenated in this order to form the observation vector. env/ and data/ must
# use the identical list, or the policy is conditioned on a permutation of what
# it was trained on.
_ARM0: List[str] = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]
_ARM1: List[str] = [
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
]

OBS_KEYS = {t: list(_ARM0) for t in ROBOMIMIC_TASKS}
OBS_KEYS["transport"] = _ARM0 + _ARM1      # two arms, both observed


# -- episode budget -----------------------------------------------------------
# Steps before a rollout is cut off. robomimic's own horizons are 400 for
# lift/can/square and 700 for transport/tool_hang.
#
# The three original tasks keep the 250 this project has always used: every
# result under outputs/ was measured at 250, and raising it would make new
# numbers incomparable with the ~30G of evaluations already on disk. Pass
# --max-steps 400 to score them on robomimic's horizon instead (a fresh sweep
# file is needed; the value is recorded in every result so the two never mix).
#
# transport and tool_hang get the full 700. They are long-horizon tasks and 250
# steps would cut them off before the task is even reachable, so there is no
# comparable history to preserve and no reason to handicap them.
MAX_STEPS = {
    "pusht": 250,
    "lift": 250,
    "can": 250,
    "square": 250,
    "transport": 700,
    "tool_hang": 700,
}


# -- perturbation target ------------------------------------------------------
# Which object the drift displaces, as a substring of its MuJoCo free-joint
# name. env.py matches these in order and takes the first joint that is present
# AND sits in the workspace, so a task can list a preferred target and a
# fallback. None means "no explicit target": take the single in-workspace free
# joint, and refuse if there is more than one.
#
# The choice matters for the result, not just for correctness: the drift is the
# whole independent variable, so perturbing the lid instead of the payload
# would measure a different experiment under the same name.
PERTURB_OBJECT = {
    "lift": None,          # one cube
    "can": None,           # one can
    "square": None,        # one nut
    # transport leaves six free joints in the workspace: payload, trash, the
    # three bins and the start-bin lid. The payload is the object the task is
    # about - the one being carried from one arm to the other - so it is the
    # analogue of the can/cube/nut, and the one whose pose going stale is what
    # this experiment measures. The bins are fixtures; drifting them would test
    # something else entirely.
    "transport": ("payload",),
    # tool_hang leaves three: stand (the base), frame (inserted into the stand
    # first) and tool (hung on the frame second). The frame is what the policy
    # picks up first and feeds into the precision-critical insertion, so it is
    # the closest analogue of the can/cube/nut and the default here.
    #
    # The alternatives measure different things and are a deliberate choice,
    # not a detail: drifting "tool" leaves phase 1 undisturbed but keeps the
    # tool moving on the table for most of the episode, and drifting "stand"
    # is the only option never exempted by the grasp-and-lift rule below, so
    # it disturbs both insertions for the whole episode.
    "tool_hang": ("frame",),
}


# -- gripper action dimensions ------------------------------------------------
# Which action dimensions carry the binary open/close command. They are ramped
# in the dataset because a step change has unbounded velocity and a flow-matching
# policy cannot represent it (see data.interpolate_binary_gripper_transitions).
#
# Single-arm tasks put it last, which is why the original code could hard-code
# -1. transport commands two arms, 7 dims each (6 OSC_POSE + 1 gripper), so it
# has two of them; smoothing only the last one would leave the first arm's
# gripper as the step the ramp exists to remove.
GRIPPER_DIMS = {t: (-1,) for t in ROBOMIMIC_TASKS}
GRIPPER_DIMS["transport"] = (6, 13)


def gripper_dims(task: str):
    if task not in GRIPPER_DIMS:
        raise ValueError(f"no gripper dims for task {task!r}; "
                         f"add them to env/robomimic/tasks.py")
    return tuple(GRIPPER_DIMS[task])


def obs_keys(task: str) -> List[str]:
    if task not in OBS_KEYS:
        raise ValueError(f"unknown robomimic task: {task!r}; "
                         f"expected one of {ROBOMIMIC_TASKS}")
    return list(OBS_KEYS[task])


def max_steps(task: str) -> int:
    if task not in MAX_STEPS:
        raise ValueError(f"no episode budget for task {task!r}; "
                         f"add one to env/robomimic/tasks.py")
    return MAX_STEPS[task]
