"""Kinematics for the user-supplied AgileX PIPER description.

This module intentionally contains no door or world coordinates. The fixed
numbers below are robot calibration data copied from piper_description: joint
origins, joint-frame rotations, limits, the mobile-base mount, and the gripper
contact offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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

# Conservative non-arm XY outline in base_link, from the actual xacro and
# BASE_FRAME / D435_BRACKET STL bounds. Includes camera housings and wheels.
# Calibration geometry, not a door/world coordinate or collision exemption.
PIPER_BODY_OUTLINE = np.array([[-.245,-.226],[-.245,.226],[.312,.226],[.312,-.226]])

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
        # The lever axis is lateral. Rolling the parallel gripper 90 degrees
        # makes its jaws close vertically around that lever while the tool axis
        # approaches along base +x. Joint zero is NOT a level tool orientation:
        # the actual chain has a built-in five-degree offset.
        self.approach_rotation = (
            _rot_y(np.pi / 2.0) @ _rot_z(float(horizontal_lever_grasp_roll_rad)))
        self._joint_origin_transforms=tuple(_transform(xyz,rpy) for xyz,rpy in _JOINT_ORIGINS)

    def contact_reach_upper_bound(self, height):
        if not math.isfinite(height):
            raise ValueError('Invalid observed contact height')
        reach = sum(np.linalg.norm(origin[0]) for origin in _JOINT_ORIGINS[1:])+self.tool_contact_offset_m
        return float(self.mount_xyz[0]+math.sqrt(max(0., reach**2-(height-self.mount_xyz[2])**2)))

    def forward(self, positions) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(positions, dtype=float)
        if joints.shape != (6,):
            raise ValueError("PIPER forward kinematics requires six joints")

        transform = _transform(self.mount_xyz)
        for origin_transform, joint_value in zip(self._joint_origin_transforms, joints):
            transform = transform @ origin_transform
            joint_rotation = np.eye(4, dtype=float)
            joint_rotation[:3, :3] = _rot_z(float(joint_value))
            transform = transform @ joint_rotation

        tool_point = transform @ np.asarray(
            (0.0, 0.0, self.tool_contact_offset_m, 1.0), dtype=float)
        return tool_point[:3], transform[:3, :3]

    def translation_condition_ratio(self, positions) -> float:
        """Smallest/largest translational Jacobian singular value (0..1)."""
        joints = np.asarray(positions, dtype=float)
        point, _ = self.forward(joints)
        step = 1.e-5
        jacobian = np.column_stack([
            (self.forward(joints + np.eye(6)[index] * step)[0] - point) / step
            for index in range(6)])
        if not np.all(np.isfinite(jacobian)):
            return 0.
        values = np.linalg.svd(jacobian, compute_uv=False)
        return float(values[-1] / values[0]) if values[0] > 1.e-9 else 0.

    def contact_velocity_barriers(self, positions, normal, panel_yaw_rate,
                                  minimum_condition=.20, joint_margin_rad=.20, gain=.25,
                                  contact_curvature=None):
        """Robot-only joint/workspace barriers on (base v, base w, tip advance)."""
        joints, normal = np.asarray(positions, float), np.asarray(normal, float)
        if (joints.shape != (6,) or normal.shape != (3,)
                or not np.all(np.isfinite(joints)) or not np.all(np.isfinite(normal))
                or abs(np.linalg.norm(normal)-1.) > 1.e-6 or abs(normal[2]) > 1.e-6
                or not np.isfinite(panel_yaw_rate) or abs(panel_yaw_rate) > .2
                or (contact_curvature is not None and not np.isfinite(contact_curvature))
                or not 0. < minimum_condition < 1. or not 0. <= joint_margin_rad < .5
                or not np.isfinite(gain) or gain <= 0.):
            raise ValueError('Invalid encoder contact workspace feedback')
        point, rotation = self.forward(joints)
        jacobian = np.empty((6, 6))
        step = 1.e-5
        for index in range(6):
            shifted_point, shifted_rotation = self.forward(joints+np.eye(6)[index]*step)
            jacobian[:3, index] = (shifted_point-point)/step
            jacobian[3:, index] = .16*self._rotation_error(rotation, shifted_rotation)/step
        inverse = jacobian.T @ np.linalg.solve(
            jacobian@jacobian.T+.025**2*np.eye(6), np.eye(6))
        task = np.zeros((6, 3))
        task[:3, 0] = [-1., 0., 0.]
        task[:3, 1] = [point[1], -point[0], 0.]
        task[:3, 2] = normal
        task[5, 1] = -.16
        if contact_curvature is not None:
            task[5, 2] = .16*contact_curvature
        offset = np.zeros(6); offset[5] = .16*panel_yaw_rate
        rates, bias = inverse@task, inverse@offset
        lower = PIPER_JOINT_LIMITS[:, 0]+joint_margin_rad
        upper = PIPER_JOINT_LIMITS[:, 1]-joint_margin_rad
        rows = [*rates, *(-rates)]
        bounds = [*(gain*(upper-joints)-bias), *(gain*(joints-lower)+bias)]
        condition = self.translation_condition_ratio(joints)
        delta = 1.e-4
        gradient = np.array([(self.translation_condition_ratio(joints+np.eye(6)[i]*delta)
                              - self.translation_condition_ratio(joints-np.eye(6)[i]*delta))/(2.*delta)
                             for i in range(6)])
        rows.append(-gradient@rates)
        bounds.append(gain*(condition-minimum_condition)+gradient@bias)
        return np.asarray(rows), np.asarray(bounds), condition

    def docking_contact_posture(self, observed, minimum_x, joint_margin_rad=.20,
                                camera_xyz=None, camera_half_fov=None,
                                observed_span_m=None, candidate_lanes=None,
                                prefer_mount_lane=False, prefer_compact=False,
                                pregrasp_clearance_m=None,press_reserve_m=None,
                                press_roll_rad=None):
        """Leave load-following reach before grasp, at observed height/lane.

        Select a bounded start from the robot chain. Contact integration can
        prioritize alignment with the arm mount; camera visibility and joint
        reserve remain mandatory for every candidate.
        """
        observed = np.asarray(observed, float)
        if (observed.shape != (3,) or not np.all(np.isfinite(observed))
                or not np.isfinite(minimum_x) or minimum_x <= 0.
                or observed[0] < minimum_x):
            raise ValueError('Invalid measured docking bounds')
        if pregrasp_clearance_m is not None and (not np.isfinite(pregrasp_clearance_m)
                or pregrasp_clearance_m <= 0.):
            raise ValueError('Invalid observed pre-grasp clearance')
        if press_reserve_m is not None and (not np.isfinite(press_reserve_m)
                or press_reserve_m<=0. or press_roll_rad is None
                or not np.isfinite(press_roll_rad)):
            raise ValueError('Invalid bounded press workspace probe')
        camera = None if camera_xyz is None else np.asarray(camera_xyz,float)
        if camera is not None:
            if (camera.shape != (3,) or not np.all(np.isfinite(camera))
                    or not np.isfinite(camera_half_fov) or not 0. < camera_half_fov < np.pi/2
                    or not np.isfinite(observed_span_m) or observed_span_m <= 0.):
                raise ValueError('Invalid observed camera/grip geometry')
            lanes = (np.linspace(camera[1], self.mount_xyz[1],17)
                     if candidate_lanes is None else candidate_lanes)
        else:
            lanes = (observed[1],)
        if prefer_mount_lane:
            # Avoid centering the chassis on the grasp when the real arm is
            # mounted laterally. The measured camera span still has to fit.
            lanes=sorted(lanes,key=lambda y:abs(y-self.mount_xyz[1]))
        if prefer_mount_lane or candidate_lanes is not None or prefer_compact:
            reach=sum(np.linalg.norm(origin[0]) for origin in _JOINT_ORIGINS[1:])+self.tool_contact_offset_m
            upper=min(observed[0],self.mount_xyz[0]+math.sqrt(max(0.,reach**2-(observed[2]-self.mount_xyz[2])**2)))
            candidates=[]
            for y in lanes:
                lower=minimum_x
                if camera is not None:
                    lower=max(lower,camera[0]+(abs(y-camera[1])+observed_span_m/2.)/np.tan(camera_half_fov))
                if lower<=upper:
                    candidates.extend((x,y) for x in np.linspace(lower,upper,17))
            if prefer_compact:
                candidates.sort(key=lambda xy:(xy[0],abs(xy[1]-self.mount_xyz[1])))
        else:
            candidates=((x,y) for x in np.linspace(minimum_x,observed[0],17) for y in lanes)
        best_press=None
        for x,y in candidates:
            goal = np.array([x,y,observed[2]])
            if camera is not None and (x <= camera[0] or
                    abs(y-camera[1])+observed_span_m/2. > (x-camera[0])*np.tan(camera_half_fov)):
                continue
            solution = self.solve(goal, max_iterations=100,
                                  position_tolerance_m=.003,
                                  minimum_joint_margin_rad=joint_margin_rad)
            if solution is None:
                continue
            if pregrasp_clearance_m is not None:
                approach=goal.copy();approach[0]-=pregrasp_clearance_m
                if self.solve(approach,max_iterations=100,position_tolerance_m=.003) is None:
                    continue
            margin = np.min(np.minimum(solution.positions-PIPER_JOINT_LIMITS[:,0],
                                       PIPER_JOINT_LIMITS[:,1]-solution.positions))
            if margin >= joint_margin_rad-1.e-9 and self.translation_condition_ratio(solution.positions) >= .25:
                if press_reserve_m is None:
                    return goal
                pressed=goal.copy();pressed[2]-=press_reserve_m
                prediction=self.solve(pressed,seed=solution.positions,
                    target_rotation=self.approach_rotation@_rot_z(press_roll_rad),
                    max_iterations=100,position_tolerance_m=.003,
                    minimum_joint_margin_rad=joint_margin_rad)
                if prediction is None:
                    continue
                press_margin=float(np.min(np.minimum(prediction.positions-PIPER_JOINT_LIMITS[:,0],
                    PIPER_JOINT_LIMITS[:,1]-prediction.positions)))
                score=(min(float(margin),press_margin),-x)
                if best_press is None or score>best_press[0]:
                    best_press=(score,goal)
        if best_press is not None:
            return best_press[1]
        raise ValueError('No measured docking pose with contact workspace reserve')

    def joint_approach_path(self, goal, start, observed_panel_x, max_joint_step=.025):
        """Avoid the fixed-wrist Cartesian workspace hole during unfolding.

        This guards limits and the tool's observed panel plane, not arbitrary
        3-D link collisions. The physical executor must still monitor tracking.
        """
        start=np.asarray(start,float)
        if (start.shape != (6,) or not np.all(np.isfinite(start))
                or not np.isfinite(observed_panel_x) or max_joint_step <= 0.
                or np.any(start < PIPER_JOINT_LIMITS[:,0]-.001)
                or np.any(start > PIPER_JOINT_LIMITS[:,1]+.001)):
            raise ValueError('Invalid observed joint approach bounds')
        # Tolerate only sub-milliradian encoder/physics residual in the seed.
        # Every commanded waypoint remains inside the unchanged URDF limits.
        start=np.clip(start,PIPER_JOINT_LIMITS[:,0],PIPER_JOINT_LIMITS[:,1])
        solution=self.solve(goal,seed=start,position_tolerance_m=.003)
        if solution is None:
            raise ValueError('Pre-grasp endpoint is outside PIPER reach')
        count=max(1,int(np.ceil(np.max(np.abs(solution.positions-start))/max_joint_step)))
        path=np.array([start+t*(solution.positions-start) for t in np.linspace(0.,1.,count+1)])
        for joints in path:
            point,rotation=self.forward(joints)
            tip=point+(self.tool_front_extent_m-self.tool_contact_offset_m)*rotation[:,2]
            if max(point[0],tip[0]) > observed_panel_x-.015:
                raise ValueError('Unfolding path would cross the observed panel plane')
        return path

    def preferred_contact_posture(self, observed, rotation, seed, minimum_x,
                                  joint_margin_rad=.20, future_rotation=None,
                                  observed_curvature=None, minimum_panel_clearance=None,
                                  steady_orientation_only=False, minimum_displacement=False):
        """Find local forward reach with manipulability, at observed grip height.

        This selects a posture for the base to track, never teleports the hand.
        The lower bound comes from the calibrated body/gripper envelope.
        """
        observed = np.asarray(observed, float)
        if (observed.shape != (3,) or not np.all(np.isfinite(observed))
                or not np.isfinite(minimum_x) or minimum_x <= 0. or minimum_x > observed[0]
                or not np.isfinite(joint_margin_rad) or not 0. <= joint_margin_rad < .5
                or (observed_curvature is not None and not np.isfinite(observed_curvature))
                or (steady_orientation_only and observed_curvature is None)
                or (minimum_panel_clearance is not None and
                    (not np.isfinite(minimum_panel_clearance) or minimum_panel_clearance <= 0.))):
            raise ValueError('Invalid observed contact posture bounds')
        maximum_reach = steady_orientation_only and not minimum_displacement
        best, best_cost = None, ((float('inf'),float('inf')) if maximum_reach else float('inf'))
        span = float(np.max(np.abs(PIPER_BODY_OUTLINE[:,1])))
        reach_bound = (sum(np.linalg.norm(origin[0]) for origin in _JOINT_ORIGINS[1:])
                       + self.tool_contact_offset_m)
        normal = np.asarray(rotation)[:,2]
        clearance = max(.03, self.tool_front_extent_m-self.tool_contact_offset_m)
        if minimum_panel_clearance is not None:
            clearance = max(clearance, minimum_panel_clearance)
        xs=np.linspace(minimum_x,observed[0],17 if steady_orientation_only else 9)
        lanes=[float(observed[1]), *np.linspace(-span,span,9),
               *np.linspace(self.mount_xyz[1]-reach_bound,self.mount_xyz[1]+reach_bound,25)]
        if steady_orientation_only:
            # The safe lane is a continuous boundary, often between grid rows.
            # Derive it from the measured curvature and calibrated footprint.
            for x in xs:
                sine=observed_curvature*x
                if 1.e-6<abs(sine)<1.:
                    n=np.array([math.sqrt(1.-sine*sine),sine])
                    lane=(clearance+max(PIPER_BODY_OUTLINE@n)-x*n[0])/n[1]
                    lanes.append(float(lane+math.copysign(.003,n[1])))
        ys=sorted(set(lanes),key=lambda y:abs(y-observed[1]))
        for y in ys:
            for x in xs:
                # Preserve measured forward grip reach in a future opening
                # stance instead of driving the chassis toward the doorway.
                reach_reference=observed[0] if steady_orientation_only else minimum_x
                cost = (((x-reach_reference)**2,(y-observed[1])**2) if maximum_reach
                        else (x-reach_reference)**2+(y-observed[1])**2)
                if cost > best_cost:
                    continue
                goal = np.array([x,y,observed[2]])
                candidate_rotation=np.asarray(rotation)
                if steady_orientation_only:
                    sine=observed_curvature*x
                    if abs(sine)>=1.:
                        continue
                    angle=math.asin(sine)-math.atan2(normal[1],normal[0])
                    c,s=math.cos(angle),math.sin(angle)
                    candidate_rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@rotation
                candidate_normal=candidate_rotation[:,2]
                if min((goal[:2]-PIPER_BODY_OUTLINE)@candidate_normal[:2]) < clearance:
                    continue
                solution = self.solve(goal, seed=seed, target_rotation=candidate_rotation,
                                      max_iterations=100, position_tolerance_m=.003,
                                      prefer_seed_solution=True)
                if solution is None:
                    continue
                margin = np.min(np.minimum(solution.positions-PIPER_JOINT_LIMITS[:,0],
                                           PIPER_JOINT_LIMITS[:,1]-solution.positions))
                if margin < joint_margin_rad:
                    continue
                predictions = ([] if future_rotation is None or steady_orientation_only
                               else [np.asarray(future_rotation)])
                if observed_curvature is not None and not steady_orientation_only:
                    sine = observed_curvature*x
                    if abs(sine) >= 1.:
                        continue
                    # For constant hand posture, w=u*n_y/x. A stable panel
                    # heading also requires w=measured door yaw rate. Their
                    # ratio is observed curvature, not a supplied hinge/radius.
                    angle = math.asin(sine)-math.atan2(normal[1],normal[0])
                    c, s = math.cos(angle), math.sin(angle)
                    yaw = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
                    predictions.append(yaw@rotation)
                future_valid = True
                for prediction in predictions:
                    future_normal = prediction[:,2]
                    if min((goal[:2]-PIPER_BODY_OUTLINE)@future_normal[:2]) < clearance:
                        future_valid = False
                        break
                    future = self.solve(goal, seed=solution.positions,
                        target_rotation=prediction, max_iterations=100,
                        position_tolerance_m=.003, prefer_seed_solution=True)
                    if future is None:
                        future_valid = False
                        break
                    future_margin = np.min(np.minimum(future.positions-PIPER_JOINT_LIMITS[:,0],
                                                     PIPER_JOINT_LIMITS[:,1]-future.positions))
                    if future_margin < joint_margin_rad:
                        future_valid = False
                        break
                if not future_valid:
                    continue
                ratio = self.translation_condition_ratio(solution.positions)
                if ratio < .20:
                    continue
                if cost < best_cost or (cost == best_cost and ratio > best[1]):
                    best, best_cost = (goal, ratio), cost
                    if maximum_reach and x == observed[0]:
                        # Lanes are ordered by lateral displacement; this is
                        # already the nearest feasible maximum-reach stance.
                        return best
        if best is None:
            raise ValueError('No reachable contact posture within calibrated bounds')
        return best

    def measured_steady_contact_posture(self, positions, rotation, curvature, clearance,
                                        panel_anchor=None, panel_normal=None):
        """Validate an attained stance, rather than a lateral soft target.

        A differential-drive chassis cannot correct lateral error at a fixed
        heading. This accepts only measured, independently feasible steady IK.
        """
        joints=np.asarray(positions,float)
        rotation=np.asarray(rotation,float)
        if (joints.shape!=(6,) or rotation.shape!=(3,3)
                or not np.all(np.isfinite(joints)) or not np.all(np.isfinite(rotation))
                or not np.isfinite(curvature) or not np.isfinite(clearance) or clearance<.03):
            raise ValueError('Invalid measured steady contact stance')
        point,_=self.forward(joints)
        sine=curvature*point[0]
        if abs(sine)>=1.:
            raise ValueError('Measured contact stance cannot sustain observed curvature')
        heading=math.atan2(rotation[1,2],rotation[0,2])
        change=math.atan2(math.sin(math.asin(sine)-heading),math.cos(math.asin(sine)-heading))
        if abs(change)>.12:
            raise ValueError('Measured stance heading is not ready')
        c,s=math.cos(change),math.sin(change)
        yaw=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        steady=yaw@rotation
        anchor,normal=point,steady[:,2]
        if panel_anchor is not None or panel_normal is not None:
            measured_anchor=np.asarray(panel_anchor,float)
            measured_normal=np.asarray(panel_normal,float)
            if (measured_anchor.shape!=(3,) or measured_normal.shape!=(3,)
                    or not np.all(np.isfinite(measured_anchor))
                    or not np.all(np.isfinite(measured_normal))
                    or abs(np.linalg.norm(measured_normal)-1.)>1.e-6
                    or abs(measured_normal[2])>.15):
                raise ValueError('Invalid independently observed steady panel geometry')
            anchor=point+yaw@(measured_anchor-point)
            normal=yaw@measured_normal
        if min((anchor[:2]-PIPER_BODY_OUTLINE)@normal[:2])<max(clearance,.0358):
            raise ValueError('Measured steady stance lacks body clearance')
        solution=self.solve(point,seed=joints,target_rotation=steady,max_iterations=100,
                            position_tolerance_m=.003,prefer_seed_solution=True)
        if solution is None or np.max(np.abs(solution.positions-joints))>.14:
            raise ValueError('Measured steady stance lacks bounded IK')
        ratios=[]
        for values in (joints,solution.positions):
            margin=np.min(np.minimum(values-PIPER_JOINT_LIMITS[:,0],PIPER_JOINT_LIMITS[:,1]-values))
            ratio=self.translation_condition_ratio(values)
            if margin<.20 or ratio<.20:
                raise ValueError('Measured steady stance lacks workspace reserve')
            ratios.append(ratio)
        return point.copy(),min(ratios)

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

    def _error(self, positions, target_xyz, orientation_weight: float,
               target_rotation=None):
        reached, rotation = self.forward(positions)
        position_error = np.asarray(target_xyz, dtype=float) - reached
        rotation_error = self._rotation_error(
            rotation, self.approach_rotation if target_rotation is None else target_rotation)
        return np.concatenate(
            (position_error, orientation_weight * rotation_error))

    def solve(
            self,
            target_xyz,
            seed=None,
            position_tolerance_m: float = 0.012,
            max_iterations: int = 480,
            orientation_tolerance_rad: float = 0.10,
            target_rotation=None,
            prefer_seed_solution: bool = False,
            minimum_joint_margin_rad: float = 0.,
            seed_only: bool = False,
            maximum_seed_step_rad: float | None = None) -> IkResult | None:
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return None
        if (not np.isfinite(orientation_tolerance_rad)
                or orientation_tolerance_rad <= 0.0):
            return None
        if not np.isfinite(minimum_joint_margin_rad) or not 0. <= minimum_joint_margin_rad < .5:
            return None
        lower=PIPER_JOINT_LIMITS[:,0]+minimum_joint_margin_rad
        upper=PIPER_JOINT_LIMITS[:,1]-minimum_joint_margin_rad
        if maximum_seed_step_rad is not None:
            reference=np.asarray(seed,float)
            if (not math.isfinite(maximum_seed_step_rad) or not 0.<maximum_seed_step_rad<=.14
                    or reference.shape!=(6,) or not np.all(np.isfinite(reference))
                    or np.any(reference<lower) or np.any(reference>upper)):
                return None
            # A contact update must stay on the current local IK branch. A
            # different global seed is not a safe escape from a small error.
            lower=np.maximum(lower,reference-maximum_seed_step_rad)
            upper=np.minimum(upper,reference+maximum_seed_step_rad)
            seed_only=True
        desired = self.approach_rotation if target_rotation is None else np.asarray(
            target_rotation, dtype=float)
        if (desired.shape != (3, 3) or not np.all(np.isfinite(desired))
                or not np.allclose(desired.T @ desired, np.eye(3), atol=1.e-6)
                or not np.isclose(np.linalg.det(desired), 1., atol=1.e-6)):
            return None

        seeds = []
        seed_reference = None
        if seed is not None:
            seed_value = np.asarray(seed, dtype=float)
            if (seed_value.shape == (6,)
                    and np.all(np.isfinite(seed_value))):
                seed_reference = np.clip(
                    seed_value,
                    lower, upper)
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

        for seed_index, initial in enumerate(seeds):
            positions = np.clip(
                np.asarray(initial, dtype=float).copy(),
                lower, upper)
            iterations = 0
            for iterations in range(max_iterations):
                error = self._error(
                    positions, target, orientation_weight, desired)
                if (np.linalg.norm(error[:3]) <= position_tolerance_m
                        and np.linalg.norm(error[3:])
                        <= orientation_weight * orientation_tolerance_rad):
                    break

                jacobian = np.empty((6, 6), dtype=float)
                for joint_index in range(6):
                    shifted = positions.copy()
                    shifted[joint_index] += finite_difference
                    shifted_error = self._error(
                        shifted, target, orientation_weight, desired)
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
                    lower, upper)

            reached, rotation = self.forward(positions)
            position_error = float(np.linalg.norm(target - reached))
            orientation_error = float(np.linalg.norm(
                self._rotation_error(rotation, desired)))
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
            # Small contact updates normally have a continuous local solution.
            # Do not evaluate five global seeds after that solution already
            # meets both original tolerances; retain fallback when it does not.
            if prefer_seed_solution and seed_reference is not None and seed_index == 0 and valid:
                return candidate
            if seed_only:
                return candidate if seed_reference is not None and valid else None
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
