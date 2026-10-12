import unittest

from fire_robot_manipulation.contact_feedback import robust_heading_rate


class ObservedHeadingRateTests(unittest.TestCase):
    def test_waits_for_measured_time_span(self):
        self.assertIsNone(robust_heading_rate([(0.,0.)]))
        self.assertIsNone(robust_heading_rate([(0.,0.),(.1,.01)]))

    def test_constant_rate(self):
        samples = [(i*.1, i*.01) for i in range(8)]
        self.assertAlmostEqual(robust_heading_rate(samples), .1)

    def test_one_camera_fk_switch_is_not_a_sustained_overspeed(self):
        samples = [(i*.1, i*.01) for i in range(8)]
        samples[4] = (.4,.14)
        self.assertAlmostEqual(robust_heading_rate(samples), .1)

    def test_sustained_excessive_rotation_still_stops(self):
        with self.assertRaises(ValueError):
            robust_heading_rate([(i*.1,i*.03) for i in range(8)])

    def test_invalid_measurements_and_settings(self):
        for samples in ([(0.,0.),(0.,.01)], [(0.,0.),(.5,float('nan'))]):
            with self.assertRaises(ValueError):
                robust_heading_rate(samples)
        for settings in ({'maximum':0.}, {'minimum_span':-1.}):
            with self.assertRaises(ValueError):
                robust_heading_rate([(0.,0.),(.5,.01)], **settings)


if __name__ == '__main__':
    unittest.main()
