import unittest
from unittest.mock import Mock
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode


class StoppedContactAnchorTests(unittest.TestCase):
    def fixture(self):
        node = ManipulationNode.__new__(ManipulationNode)
        node._feedback_full_handle_follow = True
        node._feedback_press_verified = True
        node._sim_lever_position = -.3
        node._sim_lever_initial_position = 0.
        node._sim_lever_press_min_angle = .08
        node._sim_door_position = -.4
        node._feedback_sweep_initial_door = -.4
        node._feedback_contact_orientation_reference = (0., 0., np.eye(3))
        node._feedback_config = {'press_sign': -1., 'follow_max_shift_m': .08,
                                 'tracking_limit_m': .08}
        node._feedback_sweep_state = (np.array([.5, 0., .7]),
                                     np.array([.5, 0., .7]), 0.)
        node._feedback_press_origin = np.array([.5, 0., .75])
        node._feedback_press_depth = .05
        node._feedback_follow_last_command = -np.inf
        node._manipulation_frame = 'base_link'
        node._contact_reference_frame = 'map'
        node._tf_buffer = Mock()
        node._tf_buffer.transform.side_effect = lambda message, *args, **kwargs: message
        node._feedback_tool = Mock(return_value=np.array([.495, .002, .7]))
        node._sim_time_sec = Mock(side_effect=[1., 1.2, 1.4])
        node._feedback_command_point = Mock(return_value=True)
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        return node

    def test_stopped_encoder_error_cannot_move_command_reference(self):
        node = self.fixture()
        for _ in range(3):
            self.assertTrue(node._feedback_follow_encoder_contact())
        for call in node._feedback_command_point.call_args_list:
            np.testing.assert_allclose(call.args[0], [.5, 0., .7])
        np.testing.assert_allclose(node._feedback_sweep_state[0], [.5, 0., .7])

    def test_active_probe_anchor_remains_the_next_stopped_target(self):
        node = self.fixture()
        node._feedback_sweep_state = (np.array([.504, -.003, .7]),
                                     np.array([.5, 0., .7]), 0.)
        self.assertTrue(node._feedback_follow_encoder_contact())
        np.testing.assert_allclose(node._feedback_command_point.call_args.args[0],
                                   [.504, -.003, .7])

    def test_large_measurement_jump_stops_instead_of_rebasing(self):
        node = self.fixture()
        node._feedback_tool.return_value = np.array([.3, 0., .7])
        self.assertFalse(node._feedback_follow_encoder_contact())
        node._feedback_command_point.assert_not_called()


if __name__ == '__main__':
    unittest.main()
