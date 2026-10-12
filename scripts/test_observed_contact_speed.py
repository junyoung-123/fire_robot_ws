import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import robust_contact_speed


class ObservedContactSpeedTests(unittest.TestCase):
    def test_waits_for_measured_span(self):
        self.assertIsNone(robust_contact_speed([(0.,0.,0.,.7)],[1,0,0]))

    def test_measured_velocity_projection(self):
        samples=[(i*.1,i*.004,i*.003,.7) for i in range(8)]
        self.assertAlmostEqual(robust_contact_speed(samples,[.6,.8,0.]),.048)

    def test_one_pose_correction_does_not_mean_sustained_overspeed(self):
        samples=[(i*.1,i*.004,0.,.7) for i in range(8)]
        samples[4]=(.4,.066,0.,.7)
        self.assertAlmostEqual(robust_contact_speed(samples,[1.,0.,0.]),.04)

    def test_sustained_excessive_travel_still_stops(self):
        with self.assertRaises(ValueError):
            robust_contact_speed([(i*.1,i*.02,0.,.7) for i in range(8)],[1.,0.,0.])

    def test_nonfinite_or_invalid_axis_stops(self):
        for normal in ([2.,0.,0.],[float('nan'),0.,0.]):
            with self.assertRaises(ValueError):
                robust_contact_speed([(0.,0.,0.,.7)],normal)


if __name__=='__main__':
    unittest.main()
