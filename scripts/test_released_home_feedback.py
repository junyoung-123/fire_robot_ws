import unittest
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES, PIPER_STOW


class ReleasedHomeFeedbackTests(unittest.TestCase):
    def fixture(self):
        node = object.__new__(RobotObservationManipulation)
        node._feedback_contact = True
        node._hold_handle_during_base_open = True
        node._phase = 'RETURN_HOME'
        node._feedback_config = {'max_age_sec': 1.}
        node._feedback_arm_stamp = 100.
        node._feedback_joint_msg_stamp = (10, 0)
        node._sim_arm_joint_positions = dict(zip(PIPER_JOINT_NAMES, PIPER_STOW))
        node._publish_sim_cmd_vel = Mock()
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        node._feedback_fresh = Mock(side_effect=AssertionError('Door observations not needed for stow verification'))
        node._command_sim_arm_pose = Mock(side_effect=AssertionError('Already folded; no new trajectory'))
        node._fingers_released = Mock(return_value=True)
        clock = {'wall': 100., 'sim': 10.}
        node._sim_time_sec = lambda: clock['sim']
        return node, clock

    def verify(self, node, clock, update=None):
        def sleep(_):
            clock['wall'] += .5
            clock['sim'] += .05
            if update:
                update()
        module = 'fire_robot_manipulation.robot_observation_manipulation_node'
        with patch(module+'.time.monotonic', side_effect=lambda: clock['wall']), \
                patch(module+'.time.sleep', side_effect=sleep), \
                patch(module+'.rclpy.ok', return_value=True):
            return node._move_to_home()

    def test_folded_arm_passes_without_environmental_angle_or_motion(self):
        node, clock = self.fixture()
        self.assertTrue(self.verify(node, clock))
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)
        node._feedback_fresh.assert_not_called()
        node._command_sim_arm_pose.assert_not_called()
        self.assertEqual(node._feedback_event.call_args.args, ('home_verified',))

    def test_stale_wall_receipt_fails(self):
        node, clock = self.fixture()
        node._feedback_arm_stamp = 98.
        self.assertFalse(self.verify(node, clock))
        self.assertEqual(clock['wall'], 102.)
        node._feedback_event.assert_not_called()

    def test_stale_missing_or_future_encoder_timestamp_fails(self):
        for stamp in (None, (9, 0), (20, 0)):
            with self.subTest(stamp=stamp):
                node, clock = self.fixture()
                node._feedback_joint_msg_stamp = stamp
                self.assertFalse(self.verify(node, clock))

    def test_new_original_sample_can_recover_without_refreshing_old_one(self):
        node, clock = self.fixture()
        node._feedback_joint_msg_stamp = (9, 0)
        def update():
            node._feedback_joint_msg_stamp = (10, 50000000)
            node._feedback_arm_stamp = clock['wall']
        self.assertTrue(self.verify(node, clock, update))
        self.assertEqual(node._feedback_joint_msg_stamp, (10, 50000000))

    def test_closed_fingers_fail(self):
        node, clock = self.fixture()
        node._fingers_released.return_value = False
        self.assertFalse(self.verify(node, clock))

    def test_wrong_nonfinite_or_missing_joint_fails(self):
        for value in (PIPER_STOW[0]+.071, np.nan, np.inf, None):
            with self.subTest(value=value):
                node, clock = self.fixture()
                if value is None:
                    del node._sim_arm_joint_positions[PIPER_JOINT_NAMES[0]]
                else:
                    node._sim_arm_joint_positions[PIPER_JOINT_NAMES[0]] = value
                self.assertFalse(self.verify(node, clock))
                node._feedback_event.assert_not_called()

    def test_other_sequences_keep_parent_contract(self):
        for attr, value in (('_feedback_contact', False),
                            ('_hold_handle_during_base_open', False),
                            ('_phase', 'HANDLE_HELD_BASE_OPEN')):
            node, clock = self.fixture()
            setattr(node, attr, value)
            with patch('fire_robot_manipulation.preserved_contact_policy.ManipulationNode._move_to_home',
                       return_value=False) as parent:
                self.assertFalse(self.verify(node, clock))
                parent.assert_called_once()


if __name__ == '__main__':
    unittest.main()
