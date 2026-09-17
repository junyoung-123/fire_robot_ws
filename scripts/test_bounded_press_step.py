import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import bounded_press_step


class BoundedPressStepTests(unittest.TestCase):
    def test_next_target_not_previous_target_is_bounded(self):
        actual = np.array([.7, .2, .8])
        goal = actual + [.015, .005, -.035]
        step = bounded_press_step(goal, actual, .01, .04)
        self.assertLess(step, .01)
        self.assertAlmostEqual(np.linalg.norm(goal-[0.,0.,step]-actual), .04)

    def test_clear_tracking_allows_full_increment(self):
        self.assertEqual(bounded_press_step([0.,0.,0.], [0.,0.,0.], .01, .04), .01)

    def test_jammed_or_invalid_input_fails_closed(self):
        for goal in ([0.,0.,-.04], [0.,0.,-.05], [float('nan'),0.,0.]):
            with self.assertRaises(ValueError):
                bounded_press_step(goal, [0.,0.,0.], .01, .04)


if __name__ == '__main__':
    unittest.main()
