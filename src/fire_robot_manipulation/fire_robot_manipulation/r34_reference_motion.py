"""Compare archived motion with current robot-only angle observations.

This mode is experimental only. It does not claim that the archived controller
has the newer scene/stance guards. Physical contacts remain independently
recorded, and the newer raw sensor completion certificate is still required.
"""
import math
import time

import numpy as np
import rclpy

from .contact_feedback import SimMotionDeadline
from .r34_reference import preserved_contact_policy as archived
from .r34_reference.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_NAMES, PIPER_JOINT_LIMITS


def reference_deadline(sim_now, wall_now, sim_limit, stall_limit=3., wall_limit=600.):
    # CPU slowdown may extend wall time, not motion duration or success angle.
    if sim_limit == 180.:
        wall_limit = 3000.
    return SimMotionDeadline(sim_now, wall_now, sim_limit, stall_limit, wall_limit)


archived.SimMotionDeadline = reference_deadline


class ReferenceObservationUnavailable(ValueError):
    pass


class R34ReferenceMotion:
    def _reference_opening_target(self, minimum):
        # The completion certificate tests the lower confidence bound, not
        # just the fused mean. Derive the extra motion from paired raw samples.
        now,wall=self._sim_time_sec(),time.monotonic()
        raw=[s for s in list(getattr(self,'_plane_confirmations',()))
             if 0.<=now-s['stamp']<=.5 and 0.<=wall-s['received']<=3.
             and s.get('sensor') in ('rgbd','lidar')
             and math.isfinite(s.get('uncertainty',float('nan')))
             and 0.<=s['uncertainty']<=math.radians(2.)]
        controls=list(getattr(self,'_control_plane_confirmations',()))
        if not raw or not controls:
            return minimum
        control=controls[-1]
        matched=[s for s in raw if abs(s['stamp']-control['stamp'])<1.e-5]
        if not matched:
            return minimum
        panel=matched[-1]
        bias=max(0.,abs(control['angle']-self._sim_door_initial_position)
                    -abs(panel['angle']-self._sim_door_initial_position))
        uncertainty=max(s['uncertainty'] for s in raw)
        # The grasp fusion is clipped to the raw panel confidence interval.
        # Reserve its accepted upper bound twice: once for fused overestimate,
        # once for the independent lower bound at completion. The task angle
        # and final raw-sensor certificate remain unchanged.
        confidence_bound=math.radians(2.)
        target=minimum+2.*confidence_bound
        self._feedback_event('reference_sensor_completion_target',
            task_minimum_rad=minimum,control_target_rad=target,
            independent_panel_uncertainty_rad=uncertainty,paired_fused_overestimate_rad=bias,
            qualified_confidence_bound_rad=confidence_bound,
            reserve_basis='bounded_fusion_error_plus_raw_lower_confidence_bound',
            measurement_stamp=panel['stamp'],environment_truth_used=False)
        return target

    def _feedback_tool(self):
        actual = super()._feedback_tool()
        if actual is None and getattr(self, '_r34_reference_active', False):
            # The archived arithmetic expects a vector. Never substitute an
            # old measurement when the current sensor contract fails.
            self._publish_sim_cmd_vel(0., 0.)
            raise ReferenceObservationUnavailable('R34 reference lost current tool observation')
        return actual

    def _feedback_retain_lever_press(self):
        if getattr(self, '_contact_motion_policy', 'measured') == 'r34_reference':
            if getattr(self,'_phase',None)=='HANDLE_HELD_BASE_OPEN' and not self._feedback_settle_held_target():
                return False
        # Retention uses the robot-feedback adapter's calibrated press budget
        # and synchronized held height. The frozen reference bypasses both.
        return super()._feedback_retain_lever_press()

    def _feedback_latch_clear_push(self):
        if getattr(self, '_contact_motion_policy', 'measured') != 'r34_reference':
            return super()._feedback_latch_clear_push()
        self._r34_reference_active = True
        try:
            return self._reference_sensor_timed_latch()
        except ReferenceObservationUnavailable as exc:
            return self._feedback_stop(str(exc))
        finally:
            self._publish_sim_cmd_vel(0., 0.)
            self._r34_reference_active = False

    def _reference_sensor_timed_latch(self):
        """Keep archived press/probe arithmetic with separate simulation bounds."""
        initial=self._sim_door_initial_position
        if initial is None or not self._feedback_fresh():
            return self._feedback_stop('Missing feedback before latch-clear push')
        cfg=self._feedback_config
        deadline=SimMotionDeadline(self._sim_time_sec(),time.monotonic(),30.,
            stall_limit=3.,wall_limit=600.)
        previous=abs(self._sim_door_position-initial)
        self._feedback_lever_hold_delta=cfg['press_sign']*(
            self._sim_lever_position-self._sim_lever_initial_position)
        if (previous>.03 and self._feedback_lever_hold_delta<self._sim_lever_press_min_angle):
            return self._feedback_stop('Lever contact lost as door rotated; reobservation required')
        if not self._feedback_begin_contact_follow():
            return False
        while rclpy.ok():
            failure=deadline.failure(self._sim_time_sec(),time.monotonic())
            if failure:
                return self._feedback_stop('Latch release feedback: '+failure)
            if not self._feedback_fresh():
                return self._feedback_stop('Lost fresh latch release observation')
            actual=self._feedback_tool()
            if actual is None:
                return self._feedback_stop('Lost feedback during latch-clear push')
            lever_delta=cfg['press_sign']*(self._sim_lever_position-self._sim_lever_initial_position)
            current=abs(self._sim_door_position-initial)
            if current>.03 and lever_delta<self._sim_lever_press_min_angle:
                return self._feedback_stop('Lever contact lost as door rotated; reobservation required')
            if lever_delta>=self._sim_lever_press_min_angle:
                probe=self._feedback_probe_contact_motion()
                if probe is None:
                    return False
                if probe=='relaxed':
                    continue
            current=abs(self._sim_door_position-initial)
            self._feedback_event('latch_probe',door_delta_rad=current,
                lever_delta_rad=lever_delta,commanded_depth_m=self._feedback_press_depth)
            if current>=self._sim_latch_clear_min_angle:
                self._feedback_event('latch_released',door_delta_rad=current)
                return True
            if current-previous>=.003:
                previous=current
                continue
            self._publish_sim_cmd_vel(0.,0.)
            goal=self._feedback_press_origin.copy()
            goal[2]-=self._feedback_press_depth
            actual=self._feedback_tool()
            if actual is None or np.linalg.norm(goal-actual)>cfg['tracking_limit_m']:
                return self._feedback_stop('Latch press tracking error: possible jam')
            try:
                depth=self._feedback_press_depth+archived.bounded_press_step(
                    goal,actual,min(.005,cfg['press_step_m']),cfg['tracking_limit_m'])
            except ValueError as exc:
                return self._feedback_stop(str(exc))
            if depth>cfg['press_max_travel_m']:
                return self._feedback_stop('Latch not released within press travel limit')
            goal[2]=self._feedback_press_origin[2]-depth
            if not self._feedback_command_point(goal,'feedback_latch_press'):
                return False
            lever_before=self._sim_lever_position
            door_before=abs(self._sim_door_position-initial)
            self._feedback_press_depth=depth
            self._feedback_event('latch_press_correction',commanded_depth_m=depth)
            if not self._feedback_confirm_latch_press_response(lever_before,door_before):
                return False
            self._feedback_lever_hold_delta=max(self._feedback_lever_hold_delta,
                cfg['press_sign']*(self._sim_lever_position-self._sim_lever_initial_position))
            previous=current
        return self._feedback_stop('Latch release feedback interrupted')

    def _feedback_probe_contact_motion(self, angular_z=0., speed=.02, opening_speed=None):
        if getattr(self, '_r34_reference_active', False):
            try:
                return self._reference_sensor_safe_probe(angular_z, speed, opening_speed)
            except ReferenceObservationUnavailable as exc:
                self._feedback_stop(str(exc))
                return None
        return super()._feedback_probe_contact_motion(angular_z, speed, opening_speed)

    def _reference_sensor_safe_probe(self, angular_z, speed, opening_speed):
        # Keep the archived endpoint/IK calculation. A stopped reobservation
        # invalidates this short transaction, not the real sensor clock.
        self._publish_sim_cmd_vel(0., 0.)
        if not self._feedback_retain_lever_press() or not self._feedback_follow_contact():
            return None
        if not self._feedback_settle_held_target():
            return None
        cfg = self._feedback_config
        duration = min(.35, cfg.get('follow_step_m', .008) / max(speed, .001))
        coordinated = None
        if getattr(self, '_feedback_full_handle_follow', False):
            try:
                door0, yaw0, _ = self._feedback_contact_orientation_reference
                alpha = self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
                normal = np.array([math.cos(alpha), math.sin(alpha), 0.])
                origin = self._feedback_last_cartesian_target.copy()
                for _ in range(8):
                    try:
                        endpoint, inverse_yaw = archived.coordinated_contact_endpoint(
                            origin, normal, speed, angular_z, duration,
                            cfg['follow_step_m'], opening_speed)
                        break
                    except ValueError as exc:
                        if 'exceeds bounded contact step' not in str(exc):
                            raise
                        duration *= .5
                else:
                    raise ValueError('No bounded coordinated contact duration')
                rotation = inverse_yaw @ self._feedback_contact_rotation()
                seed = self._feedback_ideal_arm_target.copy()
                solution = self._piper_kinematics.solve(
                    endpoint, seed=seed, position_tolerance_m=.001,
                    orientation_tolerance_rad=.03, target_rotation=rotation,
                    prefer_seed_solution=True)
                if solution is None:
                    raise ValueError('Coordinated contact endpoint outside arm workspace')
                end_joints = solution.positions + self._feedback_joint_bias
                if (np.any(end_joints < PIPER_JOINT_LIMITS[:, 0])
                        or np.any(end_joints > PIPER_JOINT_LIMITS[:, 1])):
                    raise ValueError('Coordinated contact would exceed PIPER limits')
                coordinated = (origin, endpoint, seed, solution.positions)
                world_angle = self._sim_door_position-door0+yaw0
                world_normal = np.array([math.cos(world_angle), math.sin(world_angle), 0.])
            except (ValueError, archived.TransformException) as exc:
                self._feedback_stop(str(exc))
                return None
        start = self._sim_time_sec()
        deadline = SimMotionDeadline(start, time.monotonic(), duration + 1.,
                                     stall_limit=2., wall_limit=10.)
        interrupted = False
        reobserved = False
        fraction = 0.
        driven = 0.
        try:
            while rclpy.ok() and self._sim_time_sec()-start < duration:
                generation = getattr(self, '_feedback_measurement_pause_generation', 0)
                if not self._feedback_fresh():
                    self._feedback_stop('Latch probe lost live joint feedback')
                    return None
                actual = self._feedback_tool()
                if actual is None:
                    self._feedback_stop('Latch probe lost live joint feedback')
                    return None
                if generation != getattr(self, '_feedback_measurement_pause_generation', 0):
                    reobserved = True
                    break
                failure = deadline.failure(self._sim_time_sec(), time.monotonic())
                if failure:
                    self._feedback_stop('Latch probe: ' + failure)
                    return None
                lever_delta = cfg['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                if lever_delta < self._sim_lever_press_min_angle:
                    self._feedback_stop('Lever contact lost during chassis probe')
                    return None
                if self._feedback_lever_hold_delta-lever_delta > cfg['lever_hold_tolerance_rad']:
                    interrupted = True
                    break
                driven = self._sim_time_sec()-start
                if coordinated is not None:
                    fraction = min(1., driven/duration)
                    origin, endpoint, seed, end = coordinated
                    ideal = seed + fraction*(end-seed)
                    goal = origin + fraction*(endpoint-origin)
                    if np.linalg.norm(goal-actual) > cfg['tracking_limit_m']:
                        self._feedback_stop('Coordinated arm/base probe tracking exceeded')
                        return None
                    joints = ideal + self._feedback_joint_bias
                    self._publish_sim_arm_targets(dict(zip(PIPER_JOINT_NAMES, joints.tolist())))
                    self._sim_last_arm_target = joints.copy()
                    self._feedback_ideal_arm_target = ideal.copy()
                    self._feedback_last_cartesian_target = goal.copy()
                    self._feedback_press_origin[:2] = goal[:2]
                self._publish_sim_cmd_vel(speed, angular_z)
                time.sleep(.025)
            if not rclpy.ok():
                self._feedback_stop('Reference contact probe interrupted')
                return None
        finally:
            self._publish_sim_cmd_vel(0., 0.)
        if coordinated is not None and opening_speed is not None:
            reference, previous, angle = self._feedback_sweep_state
            self._feedback_sweep_state = (
                reference + world_normal*opening_speed*duration*fraction, previous, angle)
        self._feedback_event('contact_probe_motion', started_sim_sec=start,
            driven_sim_sec=driven, planned_duration_sec=duration,
            interrupted_for_press=interrupted, interrupted_for_sensor_reobservation=reobserved,
            linear_x=speed, angular_z=angular_z, opening_speed_mps=opening_speed,
            execution_adapter='archived_endpoint_abort_partial_step_after_stopped_reobservation')
        return 'reobserved' if reobserved else 'relaxed' if interrupted else 'complete'

    def _feedback_handle_held_open(self, min_delta):
        if getattr(self, '_contact_motion_policy', 'measured') != 'r34_reference':
            return super()._feedback_handle_held_open(min_delta)
        original = self._piper_kinematics
        reference = PiperActualKinematics(mount_xyz=original.mount_xyz,
                                         tool_contact_offset_m=original.tool_contact_offset_m)
        joints = [self._sim_arm_joint_positions[name] for name in archived.PIPER_JOINT_NAMES]
        old_point, old_rotation = reference.forward(joints)
        new_point, new_rotation = original.forward(joints)
        if not (np.allclose(old_point, new_point, atol=1.e-10, rtol=0.)
                and np.allclose(old_rotation, new_rotation, atol=1.e-10, rtol=0.)
                and reference.tool_front_extent_m == original.tool_front_extent_m):
            return self._feedback_stop('R34 reference does not match the actual robot calibration')
        actual = self._feedback_tool()
        if actual is None or not np.all(np.isfinite(actual)):
            return self._feedback_stop('R34 handoff requires measured robot tool coordinates')
        # The integrated approach hint is not the loaded grasp. Pass the
        # observed starting posture to the unchanged archived reach search.
        previous = getattr(self, '_feedback_preferred_contact_posture', None)
        self._feedback_preferred_contact_posture = np.asarray(actual, float).copy()
        self._feedback_event('r34_reference_handoff_reanchored',
            measured_tool_xyz=self._feedback_preferred_contact_posture.tolist(),
            previous_approach_hint=None if previous is None else np.asarray(previous).tolist(),
            basis='own_encoder_FK_not_world_coordinates')
        self._feedback_event('r34_reference_comparison_started',
            scope='experimental_reference_motion_not_success_certificate',
            motion_reference='archived_R34_nominal_short_probe',
            completion='current_independent_raw_panel_observations',
            simulator_joint_input=False, original_urdf_limits=True)
        self._r34_reference_active = True
        self._piper_kinematics = reference
        try:
            reached = archived.ManipulationNode._feedback_handle_held_open(
                self, self._reference_opening_target(min_delta))
        except ReferenceObservationUnavailable as exc:
            return self._feedback_stop(str(exc))
        finally:
            self._publish_sim_cmd_vel(0., 0.)
            self._piper_kinematics = original
            self._r34_reference_active = False
        if not reached:
            return False
        deadline = SimMotionDeadline(self._sim_time_sec(), time.monotonic(), 2.,
                                     stall_limit=3., wall_limit=30.)
        while rclpy.ok():
            delta = abs(self._sim_door_position-self._sim_door_initial_position)
            if delta >= min_delta and self._feedback_opening_complete(delta, min_delta):
                self._feedback_event('handle_held_open_verified', door_delta_rad=delta,
                    scope='single_door_reference_comparison_sensor_confirmed')
                return True
            if deadline.failure(self._sim_time_sec(), time.monotonic()):
                return self._feedback_stop('R34 reference threshold lacks independent sensor confirmation')
            self._publish_sim_cmd_vel(0., 0.)
            time.sleep(.025)
        return self._feedback_stop('R34 reference confirmation interrupted')
