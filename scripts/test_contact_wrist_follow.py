import unittest
from unittest.mock import patch

import numpy as np

from fire_robot_manipulation.contact_feedback import measured_wrist_rotation, swept_contact_reference
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, _rot_z
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode


class ContactWristTests(unittest.TestCase):
    def test_approach_is_panel_normal_not_the_tilted_joint_zero_pose(self):
        kin = PiperActualKinematics()
        _, joint_zero = kin.forward(np.zeros(6))
        self.assertGreater(abs(joint_zero[2, 2]), .08)
        np.testing.assert_allclose(kin.approach_rotation[:, 2], [1., 0., 0.], atol=1.e-12)
        np.testing.assert_allclose(kin.approach_rotation[:, 0], [0., 1., 0.], atol=1.e-12)

    def test_lateral_measurement_noise_cannot_walk_contact_goal_along_handle(self):
        reference = np.array([1., .2, .8])
        previous = reference.copy()
        for _ in range(100):
            measured = reference + [0., .01, 0.]
            reference = swept_contact_reference(reference, previous, measured,
                                                [1., 0., 0.], 0., 0., .15)
            previous = measured
        np.testing.assert_allclose(reference, [1., .2, .8])

    def test_sweep_matches_circles_of_different_unknown_radius(self):
        for radius in (.35, .6, 1.):
            previous = np.array([0., radius, .8])
            current = np.array([radius*np.sin(.2), radius*np.cos(.2), .8])
            predicted = swept_contact_reference(previous, previous, current,
                                                  [1., 0., 0.], 0., -.2, .25)
            np.testing.assert_allclose(predicted, current)

    def test_sweep_is_independent_of_world_origin_and_yaw(self):
        reference = np.array([0., .6, .8])
        current = np.array([.1, .58, .8])
        expected = swept_contact_reference(reference, reference, current,
                                            [1., 0., 0.], 0., -.15, .2)
        rot = _rot_z(.7)
        offset = np.array([12., -4., 0.])
        rotated = swept_contact_reference(rot@reference+offset, rot@reference+offset,
                                           rot@current+offset, rot@np.array([1., 0., 0.]),
                                           0., -.15, .2)
        np.testing.assert_allclose(rotated, rot@expected+offset)

    def test_sweep_rejects_unbounded_or_invalid_observations(self):
        for current in ([1., 0., 0.], [float('nan'), 0., 0.]):
            with self.assertRaises(ValueError):
                swept_contact_reference([0., 0., 0.], [0., 0., 0.], current,
                                         [1., 0., 0.], 0., -.2, .15)

    def test_wrist_rotates_with_measured_door_not_world_coordinates(self):
        rotation = PiperActualKinematics().approach_rotation
        np.testing.assert_allclose(measured_wrist_rotation(rotation, -.2, 0.), _rot_z(-.2) @ rotation)

    def test_equal_base_and_door_rotation_cancel_in_base_frame(self):
        rotation = PiperActualKinematics().approach_rotation
        np.testing.assert_allclose(measured_wrist_rotation(rotation, -.2, -.2), rotation)

    def test_measured_lever_tilt_rotates_jaws_without_tilting_approach_normal(self):
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        desired = measured_wrist_rotation(rotation, -.2, .1, -.3)
        # Tool X follows the lever; tool Z remains the panel approach normal.
        np.testing.assert_allclose(desired[:, 0],
            _rot_z(-.3) @ np.array([0., np.cos(.3), -np.sin(.3)]), atol=1.e-12)
        np.testing.assert_allclose(desired[:, 2], _rot_z(-.3) @ np.array([1., 0., 0.]), atol=1.e-12)

    def test_ik_can_follow_lever_tilt_with_original_joint_limits(self):
        kin = PiperActualKinematics()
        desired = measured_wrist_rotation(kin.approach_rotation, -.15, 0., -.3)
        solution = kin.solve([.82, .15, .72], target_rotation=desired, position_tolerance_m=.003)
        self.assertIsNotNone(solution)
        point, rotation = kin.forward(solution.positions)
        self.assertLess(np.linalg.norm(point-[.82, .15, .72]), .003)
        self.assertLess(np.linalg.norm(kin._rotation_error(rotation, desired)), .1)

    def test_nonfinite_lever_rotation_is_rejected(self):
        with self.assertRaises(ValueError):
            measured_wrist_rotation(np.eye(3), 0., 0., float('nan'))

    def test_yaw_wrapping_does_not_jump_orientation(self):
        rotation = np.eye(3)
        a = measured_wrist_rotation(rotation, -.2, .1)
        b = measured_wrist_rotation(rotation, -.2, .1 - 2*np.pi)
        np.testing.assert_allclose(a, b, atol=1.e-12)

    def test_nonfinite_feedback_is_rejected(self):
        with self.assertRaises(ValueError):
            measured_wrist_rotation(np.eye(3), float('nan'), 0.)

    def test_ik_solves_measured_yaw_without_changing_arm_dimensions(self):
        kin = PiperActualKinematics()
        target = [.82, .15, .73]
        desired = measured_wrist_rotation(kin.approach_rotation, -.2, 0.)
        solution = kin.solve(target, target_rotation=desired, position_tolerance_m=.003)
        self.assertIsNotNone(solution)
        point, rotation = kin.forward(solution.positions)
        self.assertLess(np.linalg.norm(point-target), .003)
        self.assertLess(np.linalg.norm(kin._rotation_error(rotation, desired)), .1)

    def test_ik_rejects_reflection_or_invalid_rotation(self):
        kin = PiperActualKinematics()
        for rotation in (np.diag([1., 1., -1.]), np.ones((3, 3)), np.zeros((2, 2))):
            self.assertIsNone(kin.solve([.82, .15, .73], target_rotation=rotation))

    def test_fixed_pregrasp_orientation_remains_default(self):
        node = object.__new__(ManipulationNode)
        self.assertIsNone(node._feedback_contact_rotation())

    def test_encoder_follow_compensates_measured_base_yaw(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact_orientation_reference = (.03, .4, np.eye(3))
        node._sim_door_position = -.12
        with patch.object(node, '_feedback_base_yaw', return_value=.45):
            np.testing.assert_allclose(node._feedback_contact_rotation(), _rot_z(-.2))

    def test_node_uses_live_lever_delta_instead_of_a_requested_press_angle(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact_orientation_reference = (.03, .4, np.eye(3))
        node._feedback_wrist_lever_zero = .01
        node._sim_door_position = -.12
        node._sim_lever_position = -.24
        with patch.object(node, '_feedback_base_yaw', return_value=.45):
            np.testing.assert_allclose(node._feedback_contact_rotation(),
                measured_wrist_rotation(np.eye(3), -.15, .05, -.25))

    def test_stale_base_transform_is_not_replaced_with_zero_yaw(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact_orientation_reference = (0., 0., np.eye(3))
        node._sim_door_position = -.1
        with patch.object(node, '_feedback_base_yaw', side_effect=ValueError('stale')):
            with self.assertRaisesRegex(ValueError, 'stale'):
                node._feedback_contact_rotation()

    def test_wrist_press_lead_is_small_and_relative_to_measured_lever(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact_orientation_reference = (0., 0., np.eye(3))
        node._feedback_wrist_lever_zero = 0.
        node._sim_door_position = -.1
        node._sim_lever_position = -.15
        node._feedback_config = dict(press_roll_lead_rad=.035, press_sign=-1.)
        with patch.object(node, '_feedback_base_yaw', return_value=0.):
            np.testing.assert_allclose(node._feedback_contact_rotation(),
                measured_wrist_rotation(np.eye(3), -.1, 0., -.185))
            node._sim_lever_position = -.12
            np.testing.assert_allclose(node._feedback_contact_rotation(),
                measured_wrist_rotation(np.eye(3), -.1, 0., -.185))
            node._sim_lever_position = -.17
            np.testing.assert_allclose(node._feedback_contact_rotation(),
                measured_wrist_rotation(np.eye(3), -.1, 0., -.205))
            node._feedback_config['press_roll_lead_rad'] = .5
            with self.assertRaisesRegex(ValueError, 'bounded'):
                node._feedback_contact_rotation()


if __name__ == '__main__':
    unittest.main()
