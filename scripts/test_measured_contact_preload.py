import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import measured_contact_preload


class MeasuredPreloadTests(unittest.TestCase):
    def test_zero_preload_does_not_extend_arm_while_chassis_stops(self):
        actual = np.array([.6,.12,.73])
        for _ in range(1000):
            goal = measured_contact_preload(actual, [.6,-.8,0.], 0., .015)
            np.testing.assert_array_equal(goal, actual)

    def test_stationary_measured_contact_never_accumulates_commands(self):
        actual = np.array([.6,.12,.73])
        goal = actual.copy()
        for _ in range(1000):
            goal = measured_contact_preload(actual, [1,0,0], .00375, .015)
        np.testing.assert_allclose(goal, actual+[.00375,0.,0.])

    def test_bounded_preload_rotates_with_measured_panel(self):
        actual = np.array([.5,-.1,.73])
        goal = measured_contact_preload(actual, [.6,-.8,0.], .00375, .015)
        self.assertAlmostEqual(np.linalg.norm(goal-actual), .00375)

    def test_excessive_preload_is_rejected(self):
        with self.assertRaises(ValueError):
            measured_contact_preload([.5,0,.7],[1,0,0],.02,.015)


if __name__ == '__main__':
    unittest.main()
