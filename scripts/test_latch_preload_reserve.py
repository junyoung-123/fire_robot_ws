import unittest
from unittest.mock import Mock, patch

import numpy as np

from fire_robot_manipulation.minimal_observed_angle_node import MinimalObservedAngle
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class LatchPreloadReserveTests(unittest.TestCase):
    def node(self, z_error=.012):
        node = object.__new__(RobotObservationManipulation)
        node._phase = 'LATCH_CLEAR_PUSH'
        node._feedback_full_handle_follow = False
        node._feedback_config = dict(tool_tolerance_m=.015, tracking_limit_m=.04,
            follow_step_m=.008, press_sign=-1., press_max_travel_m=.065)
        node._feedback_last_cartesian_target = np.array([.70,.15,.729])
        node._feedback_tool = lambda: np.array([.70,.15,.729+z_error])
        node._feedback_fresh = lambda: True
        node._feedback_contact_orientation_reference = (0.,0.,np.eye(3))
        node._sim_door_position = 0.
        node._feedback_base_yaw = lambda: 0.
        node._measured_prepress_height = .765
        node._sim_arm_joint_positions = dict.fromkeys(PIPER_JOINT_NAMES, 0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=.02, gripper_right_joint=-.02)
        node._piper_kinematics = Mock()
        node._piper_kinematics.forward.return_value = (node._feedback_tool(), np.eye(3))
        node._piper_kinematics._rotation_error.return_value = np.zeros(3)
        node._feedback_contact_rotation = lambda: np.eye(3)
        node._sim_lever_position = -.2
        node._sim_lever_initial_position = 0.
        node._sim_lever_press_min_angle = .08
        node._feedback_press_origin = np.array([.70,.15,.765])
        node._feedback_press_depth = .036
        node._publish_sim_cmd_vel = Mock()
        node._feedback_command_point = Mock(return_value=True)
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        return node

    def test_latch_reserve_relaxes_before_full_follow_handoff(self):
        node = self.node()
        before = node._feedback_last_cartesian_target.copy()
        with patch.object(MinimalObservedAngle, '_feedback_settle_held_target', return_value=True):
            self.assertTrue(node._feedback_settle_held_target_once())
        target, stage = node._feedback_command_point.call_args.args
        self.assertEqual(stage, 'encoder_contact_follow')
        np.testing.assert_array_equal(target[:2], before[:2])
        self.assertAlmostEqual(target[2], .7335)
        self.assertAlmostEqual(node._feedback_press_depth, .0315)
        self.assertAlmostEqual(node._feedback_held_height, target[2])
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_existing_limit_excess_is_stopped_not_forgiven(self):
        for residual in (.0175, -.016, float('nan')):
            node = self.node(residual)
            with patch.object(MinimalObservedAngle, '_feedback_settle_held_target', return_value=True) as base:
                self.assertFalse(node._feedback_settle_held_target_once())
            base.assert_not_called()
            node._feedback_command_point.assert_not_called()

    def test_lost_grip_or_lever_never_unloads(self):
        for lost in ('grip', 'roll', 'stale'):
            node = self.node()
            if lost == 'grip':
                node._sim_arm_joint_positions['gripper_left_joint'] = .001
            elif lost == 'roll':
                node._sim_lever_position = -.04
            else:
                node._feedback_fresh = lambda: False
            self.assertFalse(node._feedback_settle_held_target_once())
            node._feedback_command_point.assert_not_called()


if __name__ == '__main__':
    unittest.main()
