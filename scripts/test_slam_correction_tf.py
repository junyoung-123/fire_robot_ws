import math
import unittest
from fire_robot_navigation.slam_correction_tf_node import correction


class CorrectionTest(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(correction((1., 2., .4), (1., 2., .4)), (0., 0., 0.))

    def test_translation(self):
        self.assertEqual(correction((0., 0., 0.), (-3., 0., 0.)), (3., 0., 0.))

    def test_rotation_and_translation_reconstruct_map_base(self):
        for map_pose, odom_pose in (((4., -2., 1.), (1., 3., -.5)),
                                    ((2., 3., -3.), (4., 1., 3.))):
            x, y, a = correction(map_pose, odom_pose)
            self.assertAlmostEqual(x+math.cos(a)*odom_pose[0]-math.sin(a)*odom_pose[1], map_pose[0])
            self.assertAlmostEqual(y+math.sin(a)*odom_pose[0]+math.cos(a)*odom_pose[1], map_pose[1])
            self.assertAlmostEqual(math.sin(a+odom_pose[2]), math.sin(map_pose[2]))


if __name__ == '__main__':
    unittest.main()
