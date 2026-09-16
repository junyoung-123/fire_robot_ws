"""Kinematics for the user-supplied AgileX PIPER description.

This module intentionally contains no door or world coordinates. The fixed
numbers below are robot calibration data copied from piper_description: joint
origins, joint-frame rotations, limits, the mobile-base mount, and the gripper
contact offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PIPER_JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

PIPER_JOINT_LIMITS = np.asarray([
    (-2.618, 2.168),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.0944, 2.0944),
], dtype=float)

# The supplied MoveIt configuration names the all-zero joint state "zero" and
# uses it as the controller initial state.
PIPER_HOME = np.zeros(6, dtype=float)

# Collision-clear navigation posture derived from the supplied PIPER chain.
# Its tool point is (0.168, 0.150, 0.919) m in base_link: above the chassis
# and behind the J100 front face, unlike the SRDF "zero" calibration state.
PIPER_STOW = np.asarray((0.0, 0.0, -1.0, 0.0, 1.0, 0.0), dtype=float)

_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.000088259, -0.091, 0.0), (1.5708, 0.0, 0.0)),
)


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0),
                       (0.0, c, -s),
                       (0.0, s, c)), dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, 0.0, s),
                       (0.0, 1.0, 0.0),
                       (-s, 0.0, c)), dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s, 0.0),
                       (s, c, 0.0),
                       (0.0, 0.0, 1.0)), dtype=float)


def _transform(xyz, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = _rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0])
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


@dataclass(frozen=True)
class IkResult:
    positions: np.ndarray
    reached_xyz: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


class PiperActualKinematics:
    """Numerical IK evaluated on the actual six-joint PIPER chain."""

    def __init__(
            self,
            mount_xyz=(0.163, 0.15, 0.509),
            tool_contact_offset_m: float = 0.10,
            horizontal_lever_grasp_roll_rad: float = np.pi / 2.0):
        self.mount_xyz = np.asarray(mount_xyz, dtype=float)
        self.tool_contact_offset_m = float(tool_contact_offset_m)
        # Furthest axial finger extent in piper_arm_actual.xacro: the official
        # gripper joint origin is z=0.1358; its collision box ends at that plane.
        self.tool_front_extent_m = 0.1358
        _, zero_rotation = self.forward(np.zeros(6, dtype=float))
        # The lever axis is lateral. Rolling the parallel gripper 90 degrees
        # makes its jaws close vertically around that lever while the tool axis
        # still approaches along base +x.
        self.approach_rotation = (
            zero_rotation @ _rot_z(float(horizontal_lever_grasp_roll_rad)))

    def forward(self, positions) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(positions, dtype=float)
        if joints.shape != (6,):
            raise ValueError("PIPER forward kinematics requires six joints")

        transform = _transform(self.mount_xyz)
        for (origin_xyz, origin_rpy), joint_value in zip(
                _JOINT_ORIGINS, joints):
            transform = transform @ _transform(origin_xyz, origin_rpy)
            joint_rotation = np.eye(4, dtype=float)
            joint_rotation[:3, :3] = _rot_z(float(joint_value))
            transform = transform @ joint_rotation

        tool_point = transform @ np.asarray(
            (0.0, 0.0, self.tool_contact_offset_m, 1.0), dtype=float)
        return tool_point[:3], transform[:3, :3]

    @staticmethod
    def _rotation_error(
            current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        relative = desired @ current.T
        cosine = float(np.clip(
            (np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        skew = np.asarray((
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1]), dtype=float)

        if angle < 1.0e-7:
            return 0.5 * skew
        if np.pi - angle > 1.0e-5:
            return angle * skew / (2.0 * np.sin(angle))

        # The cross-product form collapses to zero at exactly 180 degrees.
        # Recover the axis from the symmetric part instead so a flipped wrist
        # can never be accepted as correctly aligned.
        axis = np.sqrt(np.maximum(
            (np.diag(relative) + 1.0) * 0.5, 0.0))
        dominant = int(np.argmax(axis))
        if dominant == 0:
            axis[1] = np.copysign(axis[1], relative[0, 1])
            axis[2] = np.copysign(axis[2], relative[0, 2])
        elif dominant == 1:
            axis[0] = np.copysign(axis[0], relative[0, 1])
            axis[2] = np.copysign(axis[2], relative[1, 2])
        else:
            axis[0] = np.copysign(axis[0], relative[0, 2])
            axis[1] = np.copysign(axis[1], relative[1, 2])
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-8:
            return np.asarray((angle, 0.0, 0.0), dtype=float)
        return angle * axis / norm

    def _error(self, positions, target_xyz, orientation_weight: float):
        reached, rotation = self.forward(positions)
        position_error = np.asarray(target_xyz, dtype=float) - reached
        rotation_error = self._rotation_error(
            rotation, self.approach_rotation)
        return np.concatenate(
            (position_error, orientation_weight * rotation_error))

    def solve(
            self,
            target_xyz,
            seed=None,
            position_tolerance_m: float = 0.012,
            max_iterations: int = 240,
            orientation_tolerance_rad: float = 0.10) -> IkResult | None:
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return None
        if (not np.isfinite(orientation_tolerance_rad)
                or orientation_tolerance_rad <= 0.0):
            return None

        seeds = []
        seed_reference = None
        if seed is not None:
            seed_value = np.asarray(seed, dtype=float)
            if (seed_value.shape == (6,)
                    and np.all(np.isfinite(seed_value))):
                seed_reference = np.clip(
                    seed_value,
                    PIPER_JOINT_LIMITS[:, 0],
                    PIPER_JOINT_LIMITS[:, 1])
                seeds.append(seed_reference)
        seeds.extend((
            PIPER_HOME,
            np.asarray((0.0, 1.4, -1.6, 0.0, 0.2, 0.0)),
            np.asarray((0.5, 1.2, -1.8, -0.5, 0.5, 0.5)),
            np.asarray((-0.5, 1.2, -1.8, 0.5, 0.5, -0.5)),
            np.zeros(6, dtype=float),
        ))

        orientation_weight = 0.16
        damping = 0.025
        finite_difference = 1.0e-5
        best = None

        for initial in seeds:
            positions = np.clip(
                np.asarray(initial, dtype=float).copy(),
                PIPER_JOINT_LIMITS[:, 0], PIPER_JOINT_LIMITS[:, 1])
            iterations = 0
            for iterations in range(max_iterations):
                error = self._error(
                    positions, target, orientation_weight)
                if (np.linalg.norm(error[:3]) <= position_tolerance_m
                        and np.linalg.norm(error[3:])
                        <= orientation_weight * orientation_tolerance_rad):
                    break

                jacobian = np.empty((6, 6), dtype=float)
                for joint_index in range(6):
                    shifted = positions.copy()
                    shifted[joint_index] += finite_difference
                    shifted_error = self._error(
                        shifted, target, orientation_weight)
                    jacobian[:, joint_index] = (
                        shifted_error - error) / finite_difference

                system = (
                    jacobian @ jacobian.T
                    + damping * damping * np.eye(6, dtype=float))
                try:
                    delta = -jacobian.T @ np.linalg.solve(system, error)
                except np.linalg.LinAlgError:
                    break
                positions = np.clip(
                    positions + np.clip(delta, -0.14, 0.14),
                    PIPER_JOINT_LIMITS[:, 0],
                    PIPER_JOINT_LIMITS[:, 1])

            reached, rotation = self.forward(positions)
            position_error = float(np.linalg.norm(target - reached))
            orientation_error = float(np.linalg.norm(
                self._rotation_error(rotation, self.approach_rotation)))
            candidate = IkResult(
                positions=positions,
                reached_xyz=reached,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                iterations=iterations + 1)
            continuity_cost = 0.0
            best_continuity_cost = 0.0
            if seed_reference is not None:
                continuity_cost = 0.003 * float(np.linalg.norm(
                    candidate.positions - seed_reference))
                if best is not None:
                    best_continuity_cost = 0.003 * float(np.linalg.norm(
                        best.positions - seed_reference))
            valid = (candidate.position_error_m <= position_tolerance_m
                     and candidate.orientation_error_rad <= orientation_tolerance_rad)
            best_valid = (best is not None
                          and best.position_error_m <= position_tolerance_m
                          and best.orientation_error_rad <= orientation_tolerance_rad)
            if best is None or (valid and not best_valid) or (valid == best_valid and (
                    candidate.position_error_m
                    + 0.02 * candidate.orientation_error_rad
                    + continuity_cost
                    < best.position_error_m
                    + 0.02 * best.orientation_error_rad
                    + best_continuity_cost)):
                best = candidate

        if (best is None or best.position_error_m > position_tolerance_m
                or best.orientation_error_rad > orientation_tolerance_rad):
            return None
        return best
