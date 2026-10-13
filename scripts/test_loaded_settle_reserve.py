"""Measured V6 settling drift must not escape the preload guard's loop."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from scripts import test_latch_preload_reserve as fixtures


class LoadedSettleReserveTests(unittest.TestCase):
    def node(self):
        node = fixtures.LatchPreloadReserveTests().node()
        node._phase = 'HANDLE_HELD_BASE_OPEN'
        node._feedback_full_handle_follow = True
        node._feedback_last_cartesian_target = np.array([.5428526219, .0851692627, .7408596005])
        node._feedback_press_origin = np.array([.54, .085, .7637596983])
        node._feedback_press_depth = .0229000978
        node._measured_prepress_height = .7637596983
        node._sim_last_arm_target = np.zeros(6)
        node._feedback_joint_bias = np.zeros(6)
        node._feedback_sweep_state = (np.zeros(3), np.zeros(3), 0.)
        node._manipulation_frame = 'base_link'
        node._contact_reference_frame = 'contact_scan'
        node._tf_buffer = Mock()
        node._tf_buffer.transform.side_effect = lambda point, *args, **kwargs: point
        node._feedback_command_point.side_effect = lambda goal, stage: (
            setattr(node, '_feedback_last_cartesian_target', np.asarray(goal).copy()) or True)
        node._piper_kinematics._rotation_error.return_value = np.array([.05, 0., 0.])
        return node

    def test_recorded_vertical_drift_is_relieved_inside_stopped_settling(self):
        node = self.node()
        # V6 World1 third door, 1156.562--1159.562 seconds: own encoder FK only.
        observations = np.array([
            [.5398922249, .0978422978, .7483596005],
            [.5411092351, .0978891358, .7505330628],
            [.5420451137, .0986405637, .7557020961],
            [.5426229304, .0987941013, .7581148626]])
        now = [0.]
        node._sim_time_sec = lambda: now[0]
        node._feedback_tool = lambda: observations[min(int(now[0]/.5), 3)].copy()
        goal = node._feedback_last_cartesian_target.copy()
        with patch('rclpy.ok', return_value=True), patch('time.sleep', side_effect=lambda _: now.__setitem__(0, now[0]+.1)):
            result = ManipulationNode._feedback_move_tool(node, goal, 'encoder_contact_follow')
        self.assertTrue(result)
        node._feedback_stop.assert_not_called()
        releases = [c for c in node._feedback_event.call_args_list if c.args[0] == 'measured_vertical_preload_released']
        self.assertGreaterEqual(len(releases), 2)
        for event in releases:
            old = np.array(event.kwargs['previous_target_xyz'])
            new = np.array(event.kwargs['requested_tool_xyz'])
            actual = np.array(event.kwargs['actual_tool_xyz'])
            np.testing.assert_array_equal(old[:2], new[:2])
            self.assertLessEqual(abs(actual[2]-old[2]), .015)
            self.assertLessEqual(np.linalg.norm(new-old), .008)
            self.assertAlmostEqual(actual[2]-new[2], .0075)
            self.assertGreater(event.kwargs['retained_roll_rad'], .08)
        self.assertLess(now[0], 8.)
        self.assertTrue(all(c.args == (0., 0.) for c in node._publish_sim_cmd_vel.call_args_list))

    def test_already_excessive_height_is_stopped_not_retargeted(self):
        node = self.node()
        goal = node._feedback_last_cartesian_target.copy()
        self.assertIsNone(node._feedback_loaded_settle_goal(goal, goal+[0.,0.,.0173]))
        node._feedback_command_point.assert_not_called()
        node._feedback_stop.assert_called_once()

    def test_normal_excess_and_nonfinite_measurement_still_stop(self):
        for delta in ([.016,0.,.009], [0.,0.,float('nan')]):
            node = self.node()
            goal = node._feedback_last_cartesian_target.copy()
            self.assertIsNone(node._feedback_loaded_settle_goal(goal, goal+delta))
            node._feedback_command_point.assert_not_called()

    def test_no_unloading_without_grip_wrist_and_fresh_observation(self):
        for missing in ('grip', 'wrist', 'lever', 'fresh'):
            node = self.node()
            goal = node._feedback_last_cartesian_target.copy()
            if missing == 'grip':
                node._sim_arm_joint_positions['gripper_left_joint'] = 0.
            elif missing == 'wrist':
                node._piper_kinematics._rotation_error.return_value = np.array([.2,0.,0.])
            elif missing == 'lever':
                node._sim_lever_position = -.03
            else:
                node._feedback_fresh = lambda: False
            self.assertIsNone(node._feedback_loaded_settle_goal(goal, goal+[0.,0.,.01]))
            node._feedback_command_point.assert_not_called()

    def test_no_unloading_above_measured_prepress_height(self):
        node = self.node()
        goal = np.array([.54,.085,.756])
        np.testing.assert_array_equal(node._feedback_loaded_settle_goal(goal, goal+[0.,0.,.01]), goal)
        node._feedback_command_point.assert_not_called()

    def test_legacy_hook_and_other_phase_do_not_change_target(self):
        goal = np.array([.5,.1,.7])
        self.assertIs(ManipulationNode._feedback_loaded_settle_goal(SimpleNamespace(), goal, goal), goal)
        node = self.node()
        node._phase = 'PRE_GRASP'
        self.assertIs(node._feedback_loaded_settle_goal(goal, goal+[0.,0.,.1]), goal)
        node._feedback_command_point.assert_not_called()


if __name__ == '__main__':
    unittest.main()
