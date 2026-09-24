"""Rotation representations for absolute end-effector actions.

robosuite's OSC controller in absolute mode takes [pos(3), axis-angle(3)] per
arm, and that is also what the dataset conversion stores. The policy does not
train on axis-angle, though: the Panda's default gripper-down pose is a ~180
degree rotation, and axis-angle wraps at pi, so a trajectory that hovers there
flips sign between consecutive steps and hands the flow a discontinuity it
cannot integrate. Following Diffusion Policy, the policy sees the first two rows
of the rotation matrix (the "6D" representation, continuous everywhere) and the
wrapper converts back to axis-angle at env.step.

Conventions: quaternions from robosuite are (x, y, z, w). rot6d is
R[0, :] ++ R[1, :]; the inverse Gram-Schmidts the two rows and completes the
frame with their cross product.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_matrix(q):
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()


def rotvec_to_matrix(rv):
    return Rotation.from_rotvec(np.asarray(rv, dtype=np.float64)).as_matrix()


def matrix_to_rotvec(m):
    return Rotation.from_matrix(np.asarray(m, dtype=np.float64)).as_rotvec()


def matrix_to_rot6d(m):
    m = np.asarray(m, dtype=np.float64)
    return np.concatenate([m[..., 0, :], m[..., 1, :]], axis=-1)


def rot6d_to_matrix(d6):
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def abs7_to_abs10(a7):
    """[pos3, rotvec3, grip1] (x n arms, flattened) -> [pos3, rot6d, grip1]."""
    a7 = np.asarray(a7, dtype=np.float64)
    lead = a7.shape[:-1]
    arms = a7.reshape(*lead, -1, 7)
    r6 = matrix_to_rot6d(rotvec_to_matrix(arms[..., 3:6].reshape(-1, 3))).reshape(*arms.shape[:-1], 6)
    return np.concatenate([arms[..., :3], r6, arms[..., 6:7]], axis=-1).reshape(*lead, -1)


def abs10_to_abs7(a10):
    """Inverse of abs7_to_abs10, for handing the action to robosuite."""
    a10 = np.asarray(a10, dtype=np.float64)
    lead = a10.shape[:-1]
    arms = a10.reshape(*lead, -1, 10)
    rv = matrix_to_rotvec(rot6d_to_matrix(arms[..., 3:9].reshape(-1, 6))).reshape(*arms.shape[:-1], 3)
    return np.concatenate([arms[..., :3], rv, arms[..., 9:10]], axis=-1).reshape(*lead, -1)
