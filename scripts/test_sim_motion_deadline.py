import unittest

from fire_robot_manipulation.contact_feedback import SimMotionDeadline


class SimMotionDeadlineTests(unittest.TestCase):
    def test_slow_but_progressing_sim_does_not_spend_wall_motion_budget(self):
        guard = SimMotionDeadline(0., 0., 90.)
        for wall in range(1, 301):
            self.assertIsNone(guard.failure(wall * .15, float(wall)))

    def test_stalled_clock_stops_even_before_motion_budget(self):
        guard = SimMotionDeadline(0., 0., 90.)
        self.assertIn('stalled', guard.failure(0., 3.))

    def test_simulation_deadline_is_finite(self):
        guard = SimMotionDeadline(10., 0., 90.)
        self.assertIn('Simulation motion', guard.failure(100., 1.))

    def test_infinite_slow_progress_has_absolute_cap(self):
        guard = SimMotionDeadline(0., 0., 90., wall_limit=10.)
        for wall in range(1, 10):
            self.assertIsNone(guard.failure(wall * .01, float(wall)))
        self.assertIn('Absolute', guard.failure(.1, 10.))

    def test_clock_jump_is_not_a_new_budget(self):
        guard = SimMotionDeadline(10., 10., 90.)
        self.assertIn('backwards', guard.failure(9., 11.))

    def test_nonfinite_clock_and_invalid_limits_fail(self):
        guard = SimMotionDeadline(0., 0., 90.)
        self.assertIn('Non-finite', guard.failure(float('nan'), 1.))
        with self.assertRaises(ValueError):
            SimMotionDeadline(0., 0., 0.)


if __name__ == '__main__':
    unittest.main()
