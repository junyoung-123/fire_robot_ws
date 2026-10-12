import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import observed_press_increment


class ObservedPressBudgetTests(unittest.TestCase):
    def test_next_target_stays_within_pose_tolerance(self):
        actual = np.array([.69, .245, .756])
        goal = actual.copy()
        for _ in range(30):
            increment = observed_press_increment(goal, actual, .01, .04, .015)
            goal[2] -= increment
            self.assertLess(np.linalg.norm(goal-actual), .015)
        self.assertEqual(increment, 0.)

    def test_measured_response_creates_new_room(self):
        goal = np.array([.69, .245, .741])
        actual = goal+np.array([0., 0., .014])
        before = observed_press_increment(goal, actual, .01, .04, .015)
        actual[2] -= .004
        after = observed_press_increment(goal, actual, .01, .04, .015)
        self.assertGreater(after, before)
        self.assertLessEqual(after, .01)

    def test_excessive_recorded_preload_is_rejected(self):
        goal = np.array([.6818, .2538, .720962])
        actual = np.array([.6830, .2537, .736628])
        with self.assertRaises(ValueError):
            observed_press_increment(goal, actual, .005, .04, .015)

    def test_lateral_load_reduces_press_budget(self):
        goal = np.array([.69, .245, .74])
        actual = goal+np.array([.01, 0., .005])
        increment = observed_press_increment(goal, actual, .01, .04, .015)
        next_goal = goal.copy()
        next_goal[2] -= increment
        self.assertLess(np.linalg.norm(next_goal-actual), .015)

    def test_invalid_limits_rejected(self):
        for tolerance in (0., .05, float('nan')):
            with self.assertRaises(ValueError):
                observed_press_increment([0., 0., 0.], [0., 0., 0.], .01, .04, tolerance)


if __name__ == '__main__':
    unittest.main()
