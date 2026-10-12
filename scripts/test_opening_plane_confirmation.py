import unittest
from fire_robot_manipulation.contact_feedback import confirmed_plane_opening


class OpeningPlaneConfirmationTests(unittest.TestCase):
    def samples(self,source='lidar',angle=-2.1,uncertainty=.02):
        return [dict(stamp=10.+i*.1,received=100.+i*.1,angle=angle,
                     uncertainty=uncertainty,sensor=source) for i in range(3)]

    def valid(self,samples,sim_now=10.3,wall_now=100.3):
        return confirmed_plane_opening(samples,0.,2.05,sim_now,wall_now)

    def test_fresh_lidar_geometry_can_confirm_when_rgbd_is_occluded(self):
        self.assertIsNotNone(self.valid(self.samples()))

    def test_rgbd_geometry_can_confirm(self):
        self.assertIsNotNone(self.valid(self.samples('rgbd')))

    def test_mixed_high_rate_sensors_are_checked_over_time_not_last_three_frames(self):
        samples=[dict(stamp=10.+i*.025,received=100.+i*.025,angle=-2.1,
                      uncertainty=.02,sensor='rgbd' if i%2 else 'lidar') for i in range(12)]
        self.assertIsNotNone(self.valid(samples))

    def test_uncertainty_must_clear_the_same_opening_threshold(self):
        self.assertIsNone(self.valid(self.samples(angle=-2.06)))

    def test_one_or_duplicated_measurement_cannot_confirm(self):
        self.assertIsNone(self.valid(self.samples()[:1]*3))

    def test_stale_future_or_missing_quality_cannot_confirm(self):
        self.assertIsNone(self.valid(self.samples(),sim_now=11.))
        self.assertIsNone(self.valid(self.samples(),wall_now=104.))
        self.assertIsNone(self.valid(self.samples(),sim_now=9.))
        self.assertIsNone(self.valid(self.samples(uncertainty=float('nan'))))

    def test_truth_or_encoder_inference_is_not_geometric_confirmation(self):
        for source in ('gazebo','joint_state','encoder','controller_target'):
            self.assertIsNone(self.valid(self.samples(source)))


if __name__=='__main__':
    unittest.main()
