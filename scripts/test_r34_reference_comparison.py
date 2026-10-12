import hashlib
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_NAMES, PIPER_JOINT_LIMITS
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.minimal_observed_angle_node import MinimalObservedAngle
from fire_robot_manipulation import r34_reference_motion as reference


class R34ReferenceComparisonTests(unittest.TestCase):
    def test_archived_sources_are_byte_identical_to_preserved_r34(self):
        folder = Path(reference.archived.__file__).parent
        hashes = {
            'preserved_contact_policy.py': 'ea9a63ec1d7c26fff740d62208abf9e0d9ec01cb0c83b2ee530a9c185b85ac0a',
            'piper_actual_kinematics.py': 'e7822f02c76ae8e29e409220974cd6b6572c3cadef868990d58c3865fe9f08d3',
            'contact_feedback.py': 'dcb991056ca41ff3f1d0ebbd76caf386bfc867124163c81ca92c7714c2cc7b47',
        }
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(), digest)

    def test_actual_urdf_fk_and_limits_match_archived_robot(self):
        current = PiperActualKinematics()
        old = reference.PiperActualKinematics(mount_xyz=current.mount_xyz,
                                             tool_contact_offset_m=current.tool_contact_offset_m)
        rng = np.random.default_rng(171)
        for _ in range(25):
            joints = rng.uniform(PIPER_JOINT_LIMITS[:, 0], PIPER_JOINT_LIMITS[:, 1])
            for actual, expected in zip(current.forward(joints), old.forward(joints)):
                np.testing.assert_allclose(actual, expected, atol=1.e-10, rtol=0.)
        np.testing.assert_array_equal(PIPER_JOINT_LIMITS, reference.archived.PIPER_JOINT_LIMITS)

    def node(self):
        node = RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._contact_motion_policy = 'r34_reference'
        node._piper_kinematics = PiperActualKinematics()
        node._sim_arm_joint_positions = dict(zip(PIPER_JOINT_NAMES, [.2, 1., -1.5, .1, .2, .1]))
        node._feedback_tool = lambda: node._piper_kinematics.forward(
            [node._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])[0]
        node._publish_sim_cmd_vel = Mock()
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        node._sim_time_sec = lambda: 10.
        node._sim_door_position, node._sim_door_initial_position = 2.1, 0.
        return node

    def test_reference_threshold_is_not_sufficient_without_raw_sensor_certificate(self):
        node = self.node()
        original = node._piper_kinematics
        node._feedback_opening_complete = Mock(return_value=False)
        with patch.object(reference.archived.ManipulationNode, '_feedback_handle_held_open', return_value=True), \
                patch.object(reference.rclpy, 'ok', return_value=True), \
                patch.object(reference, 'SimMotionDeadline') as deadline:
            deadline.return_value.failure.return_value = 'bounded_reconfirmation_timeout'
            self.assertFalse(node._feedback_handle_held_open(2.05))
        self.assertIs(node._piper_kinematics, original)
        self.assertFalse(node._r34_reference_active)
        node._feedback_opening_complete.assert_called_once_with(2.1, 2.05)
        self.assertFalse(any(call.args[0] == 'handle_held_open_verified' for call in node._feedback_event.call_args_list))

    def test_independent_raw_certificate_can_confirm_reference_threshold(self):
        node = self.node()
        node._feedback_opening_complete = Mock(return_value=True)
        with patch.object(reference.archived.ManipulationNode, '_feedback_handle_held_open', return_value=True), \
                patch.object(reference.rclpy, 'ok', return_value=True):
            self.assertTrue(node._feedback_handle_held_open(2.05))
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)
        self.assertEqual(node._feedback_event.call_args.args[0], 'handle_held_open_verified')

    def test_failure_and_exception_restore_experimental_model_and_stop_chassis(self):
        for failure in (False, RuntimeError('test_failure')):
            node = self.node()
            original = node._piper_kinematics
            with patch.object(reference.archived.ManipulationNode, '_feedback_handle_held_open',
                              **({'return_value': failure} if failure is False else {'side_effect': failure})):
                if failure is False:
                    self.assertFalse(node._feedback_handle_held_open(2.05))
                else:
                    with self.assertRaises(RuntimeError):
                        node._feedback_handle_held_open(2.05)
            self.assertIs(node._piper_kinematics, original)
            self.assertFalse(node._r34_reference_active)
            node._publish_sim_cmd_vel.assert_called_with(0., 0.)

    def test_reference_probe_uses_sensor_safe_adapter_not_new_measured_probe(self):
        node = self.node()
        node._r34_reference_active = True
        with patch.object(node, '_reference_sensor_safe_probe', return_value='complete') as probe:
            self.assertEqual(node._feedback_probe_contact_motion(.03, .02, .012), 'complete')
        probe.assert_called_once_with(.03, .02, .012)

    def probe_node(self):
        node = self.node()
        node._r34_reference_active = True
        node._feedback_full_handle_follow = True
        node._feedback_config = dict(follow_step_m=.008, tracking_limit_m=.04,
                                     press_sign=1., lever_hold_tolerance_rad=.08)
        node._feedback_retain_lever_press = Mock(return_value=True)
        node._feedback_follow_contact = Mock(return_value=True)
        node._feedback_settle_held_target = Mock(return_value=True)
        node._feedback_base_yaw = lambda: 0.
        node._feedback_contact_rotation = lambda: np.eye(3)
        node._feedback_contact_orientation_reference = (0., 0., None)
        node._sim_door_position = .5
        node._sim_lever_position = .2
        node._sim_lever_initial_position = 0.
        node._sim_lever_press_min_angle = .1
        node._feedback_lever_hold_delta = .2
        node._feedback_last_cartesian_target = np.array([.5, 0., .74])
        node._feedback_press_origin = np.array([.5, 0., .76])
        node._feedback_tool = lambda: np.array([.5, 0., .74])
        node._feedback_ideal_arm_target = np.array([0., 1., -1.5, 0., 0., 0.])
        node._feedback_joint_bias = np.zeros(6)
        node._piper_kinematics = Mock()
        node._piper_kinematics.solve.return_value.positions = node._feedback_ideal_arm_target + .001
        node._publish_sim_arm_targets = Mock()
        node._feedback_sweep_state = (np.zeros(3), np.zeros(3), .5)
        node._feedback_measurement_pause_generation = 0
        node.clock = 10.
        node._sim_time_sec = lambda: node.clock
        return node

    def test_reobservation_aborts_partial_probe_before_deadline_or_old_endpoint(self):
        node = self.probe_node()
        calls = 0
        def fresh():
            nonlocal calls
            calls += 1
            if calls == 3:
                node._feedback_measurement_pause_generation += 1
                node.clock += 1.5
            return True
        node._feedback_fresh = fresh
        with patch.object(reference.rclpy, 'ok', return_value=True), \
                patch.object(reference.time, 'sleep', side_effect=lambda _: setattr(node, 'clock', node.clock+.05)), \
                patch.object(reference, 'SimMotionDeadline') as deadline:
            deadline.return_value.failure.return_value = None
            self.assertEqual(node._feedback_probe_contact_motion(0., .02, .012), 'reobserved')
        self.assertEqual(deadline.return_value.failure.call_count, 2)
        self.assertEqual(node._publish_sim_arm_targets.call_count, 2)
        self.assertLess(np.linalg.norm(node._feedback_last_cartesian_target-[.5, 0., .74]), .002)
        np.testing.assert_allclose(node._feedback_sweep_state[0],
            np.array([np.cos(.5), np.sin(.5), 0.])*.012*.05)
        node._feedback_stop.assert_not_called()
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)
        payload = node._feedback_event.call_args.kwargs
        self.assertTrue(payload['interrupted_for_sensor_reobservation'])
        self.assertAlmostEqual(payload['driven_sim_sec'], .05)

    def test_unrecoverable_probe_observation_never_dispatches_motion(self):
        node = self.probe_node()
        node._feedback_fresh = Mock(return_value=False)
        with patch.object(reference.rclpy, 'ok', return_value=True):
            self.assertIsNone(node._feedback_probe_contact_motion(0., .02, .012))
        node._publish_sim_arm_targets.assert_not_called()
        self.assertTrue(all(call.args == (0., 0.) for call in node._publish_sim_cmd_vel.call_args_list))
        node._feedback_stop.assert_called_once_with('Latch probe lost live joint feedback')

    def test_probe_does_not_loosen_joint_limits(self):
        node = self.probe_node()
        node._piper_kinematics.solve.return_value.positions = PIPER_JOINT_LIMITS[:, 1]+.01
        self.assertIsNone(node._feedback_probe_contact_motion(0., .02, .012))
        node._publish_sim_arm_targets.assert_not_called()
        node._feedback_stop.assert_called_once_with('Coordinated contact would exceed PIPER limits')

    def test_completion_target_uses_paired_sensor_uncertainty_without_lowering_minimum(self):
        node=self.node()
        node._sim_door_initial_position=0.
        node._plane_confirmations=[dict(stamp=9.9,received=100.,angle=.3,uncertainty=.02,sensor='rgbd')]
        node._control_plane_confirmations=[dict(stamp=9.9,angle=.306)]
        with patch.object(reference.time,'monotonic',return_value=100.):
            self.assertAlmostEqual(node._reference_opening_target(2.05),2.05+2.*np.deg2rad(2.))
        self.assertFalse(node._feedback_event.call_args.kwargs['environment_truth_used'])

    def test_completion_reserve_covers_later_qualified_confidence_and_fusion_error(self):
        node=self.node()
        node._sim_door_initial_position=0.
        node._plane_confirmations=[dict(stamp=9.9,received=100.,angle=.3,uncertainty=.01,sensor='rgbd')]
        node._control_plane_confirmations=[dict(stamp=9.9,angle=.304)]
        with patch.object(reference.time,'monotonic',return_value=100.):
            target=node._reference_opening_target(2.05)
        later_sigma=np.deg2rad(2.)
        raw_lower_bound=target-later_sigma-later_sigma
        self.assertGreaterEqual(raw_lower_bound,2.05-1.e-12)

    def test_unpaired_stale_or_future_panel_does_not_invent_target_margin(self):
        for stamp,control_stamp in ((9.,9.),(10.1,10.1),(9.9,9.8)):
            node=self.node()
            node._plane_confirmations=[dict(stamp=stamp,received=100.,angle=.3,uncertainty=.02,sensor='rgbd')]
            node._control_plane_confirmations=[dict(stamp=control_stamp,angle=.306)]
            with patch.object(reference.time,'monotonic',return_value=100.):
                self.assertEqual(node._reference_opening_target(2.05),2.05)

    def test_wall_budget_does_not_change_probe_or_motion_duration(self):
        with patch.object(reference, 'SimMotionDeadline') as factory:
            reference.reference_deadline(0., 0., 180., wall_limit=600.)
            factory.assert_called_with(0., 0., 180., 3., 3000.)
            reference.reference_deadline(0., 0., 1.35, stall_limit=2., wall_limit=10.)
            factory.assert_called_with(0., 0., 1.35, 2., 10.)

    def test_latch_and_hold_correction_use_the_same_archived_sequence(self):
        node = self.node()
        with patch.object(reference.archived.ManipulationNode, '_feedback_retain_lever_press', return_value=True) as hold, \
                patch.object(reference.archived.ManipulationNode, '_feedback_latch_clear_push', return_value=True) as latch:
            self.assertTrue(node._feedback_retain_lever_press())
            self.assertTrue(node._feedback_latch_clear_push())
        hold.assert_called_once_with(node)
        latch.assert_called_once_with(node)
        self.assertFalse(node._r34_reference_active)
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)

    def test_latch_exception_stops_and_restores_comparison_state(self):
        node = self.node()
        with patch.object(reference.archived.ManipulationNode, '_feedback_latch_clear_push', side_effect=RuntimeError('test_failure')):
            with self.assertRaises(RuntimeError):
                node._feedback_latch_clear_push()
        self.assertFalse(node._r34_reference_active)
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)

    def test_handoff_uses_measured_grasp_not_the_integrated_approach_hint(self):
        node = self.node()
        node._feedback_preferred_contact_posture = np.array([.3, -.2, .8])
        actual = node._feedback_tool().copy()
        def check_reference(subject, minimum):
            np.testing.assert_array_equal(subject._feedback_preferred_contact_posture, actual)
            self.assertEqual(minimum, 2.05)
            return False
        with patch.object(reference.archived.ManipulationNode, '_feedback_handle_held_open', side_effect=check_reference):
            self.assertFalse(node._feedback_handle_held_open(2.05))

    def test_missing_tool_cannot_be_replaced_by_an_old_hint(self):
        node = self.node()
        node._feedback_preferred_contact_posture = np.array([.6, .1, .8])
        node._feedback_tool = lambda: None
        with patch.object(reference.archived.ManipulationNode, '_feedback_handle_held_open') as handler:
            self.assertFalse(node._feedback_handle_held_open(2.05))
        handler.assert_not_called()

    def test_lost_observation_raises_before_archived_numeric_arithmetic(self):
        node = self.node()
        del node._feedback_tool
        node._r34_reference_active = True
        with patch.object(MinimalObservedAngle, '_feedback_tool', return_value=None):
            with self.assertRaises(reference.ReferenceObservationUnavailable):
                node._feedback_tool()
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)

    def test_measured_policy_keeps_its_existing_missing_observation_contract(self):
        node = self.node()
        del node._feedback_tool
        node._r34_reference_active = False
        with patch.object(MinimalObservedAngle, '_feedback_tool', return_value=None):
            self.assertIsNone(node._feedback_tool())

    def test_probe_observation_loss_returns_failure_not_node_exception(self):
        node = self.node()
        node._r34_reference_active = True
        with patch.object(node, '_reference_sensor_safe_probe',
                          side_effect=reference.ReferenceObservationUnavailable('missing')):
            self.assertIsNone(node._feedback_probe_contact_motion(.03, .02, .012))
        node._feedback_stop.assert_called_once_with('missing')

    def test_opening_observation_loss_restores_model_and_reports_failure(self):
        node = self.node()
        original = node._piper_kinematics
        with patch.object(reference.archived.ManipulationNode, '_feedback_handle_held_open',
                          side_effect=reference.ReferenceObservationUnavailable('missing')):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        self.assertIs(node._piper_kinematics, original)
        self.assertFalse(node._r34_reference_active)
        node._feedback_stop.assert_called_once_with('missing')
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)


if __name__ == '__main__':
    unittest.main()
