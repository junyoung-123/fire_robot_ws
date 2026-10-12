import math
import unittest
import numpy as np
from fire_robot_manipulation.static_line_yaw import StaticLineYaw, measured_lines
from fire_robot_manipulation.contact_scan_registration import transform_cloud,inverse_pose


class StaticLineYawTests(unittest.TestCase):
    def setUp(self):
        x=np.linspace(-3.,4.,200)
        self.wall=np.column_stack((x,np.full_like(x,2.)))
        self.reference=StaticLineYaw(self.wall)

    def test_yaw_is_observed_despite_untrusted_translation(self):
        pose=np.array([.18,-.13,.018])
        cloud=transform_cloud(self.wall,inverse_pose(pose))
        fit=self.reference.estimate(cloud,.003)
        self.assertAlmostEqual(fit['yaw'],pose[2],places=6)
        self.assertFalse(fit['translation_observed'])

    def test_noisy_measured_wall_and_short_moving_panel(self):
        rng=np.random.default_rng(915)
        pose=np.array([.19,-.08,-.015])
        wall=transform_cloud(self.wall,inverse_pose(pose))+rng.normal(0.,.035,self.wall.shape)
        door=np.column_stack((np.full(60,.5),np.linspace(-.4,.4,60)))
        fit=self.reference.estimate(np.vstack((wall,door)),0.)
        self.assertLess(abs(fit['yaw']-pose[2]),math.radians(.5))
        self.assertLess(fit['yaw_uncertainty_rad'],math.radians(1.5))

    def test_isolated_short_panel_is_not_static_heading_support(self):
        door=np.column_stack((np.full(100,.5),np.linspace(-.4,.4,100)))
        with self.assertRaisesRegex(ValueError,'No continuous'):
            self.reference.estimate(door,0.)

    def test_missing_reference_and_nonfinite_points_fail_closed(self):
        self.assertEqual(measured_lines(np.zeros((12,2))),[])
        with self.assertRaises(ValueError):
            StaticLineYaw(np.full((80,2),np.nan))
        with self.assertRaises(ValueError):
            StaticLineYaw(np.zeros((12,2))).estimate(self.wall,0.)

    def test_yaw_discontinuity_cannot_be_relaxed_by_fallback(self):
        cloud=transform_cloud(self.wall,inverse_pose([0.,0.,math.radians(4.)]))
        with self.assertRaisesRegex(ValueError,'No continuous'):
            self.reference.estimate(cloud,0.)

    def test_unoriented_axis_wrap_does_not_flip_robot_heading(self):
        for yaw in (math.pi-.01,-math.pi+.01):
            cloud=transform_cloud(self.wall,inverse_pose([0.,0.,yaw]))
            fit=self.reference.estimate(cloud,yaw+.005)
            error=math.atan2(math.sin(fit['yaw']-yaw),math.cos(fit['yaw']-yaw))
            self.assertAlmostEqual(error,0.,places=6)


if __name__=='__main__':
    unittest.main()
