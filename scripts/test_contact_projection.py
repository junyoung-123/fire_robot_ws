import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import projected_contact_increment
from fire_robot_manipulation.piper_actual_kinematics import _rot_z


class ContactProjectionTests(unittest.TestCase):
    def test_exact_circular_chords_through_full_rotation_without_known_radius(self):
        for radius in (.4, .65, 1.):
            previous = np.array([0., radius, .8])
            reference = previous.copy()
            for old, new in zip(np.linspace(0., -2.1, 101)[:-1], np.linspace(0., -2.1, 101)[1:]):
                current = _rot_z(new) @ np.array([0., radius, .8])
                midpoint = (old+new)/2.
                tangent = [np.cos(midpoint), np.sin(midpoint), 0.]
                reference = projected_contact_increment(reference, previous, current, tangent, .15)
                np.testing.assert_allclose(reference, current, atol=1.e-12)
                previous = current

    def test_no_singularity_at_right_angle(self):
        expected = np.array([.4, -.05, .8])
        result = projected_contact_increment([.4,0.,.8], [.4,0.,.8], expected, [0.,-1.,0.], .15)
        np.testing.assert_allclose(result, expected)

    def test_translation_and_world_yaw_do_not_change_the_prediction(self):
        reference, previous, actual = np.array([[.3,.4,.8], [.3,.4,.8], [.32,.39,.8]])
        axis = np.array([.8,-.6,0.])
        expected = projected_contact_increment(reference, previous, actual, axis, .15)
        rotation, offset = _rot_z(1.7), np.array([8.,-3.,2.])
        transformed = projected_contact_increment(rotation@reference+offset,
            rotation@previous+offset, rotation@actual+offset, rotation@axis, .15)
        np.testing.assert_allclose(transformed, rotation@expected+offset)

    def test_orthogonal_measurement_error_does_not_walk_reference(self):
        result = projected_contact_increment([1.,2.,.8], [1.,2.,.8], [1.,2.01,.8], [1.,0.,0.], .15)
        np.testing.assert_allclose(result, [1.,2.,.8])

    def test_large_or_nonfinite_steps_still_rejected(self):
        for point in ([1.,0.,0.], [float('nan'),0.,0.]):
            with self.assertRaises(ValueError):
                projected_contact_increment([0.,0.,0.], [0.,0.,0.], point, [1.,0.,0.], .15)


if __name__ == '__main__':
    unittest.main()
