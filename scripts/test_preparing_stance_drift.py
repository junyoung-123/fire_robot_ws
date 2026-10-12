import math
import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import (
    contact_posture_command, chassis_supplied_opening_speed, scanned_body_constraints)
from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE


class PreparingStanceDriftTests(unittest.TestCase):
    def test_close_fixed_obstacle_does_not_allow_growing_arm_posture_error(self):
        point = np.array([.6255, .212, .7404])
        preferred = np.array([.5235, -.219, .7404])
        normal = np.array([math.cos(-.3), math.sin(-.3), 0.])
        scene = scanned_body_constraints([[.43, -.22], [.45, -.25], [.5, -.3]],
                                         PIPER_BODY_OUTLINE, margin=.12)[:2]
        v, w, _, _ = contact_posture_command(point, normal, preferred, PIPER_BODY_OUTLINE,
            margin=.08, opening_speed=.03, observed_curvature=-1.67,
            heading_priority_weight=1./1.67**2, measured_heading_reference=True,
            allow_posture_recovery=True, limit_posture_drift=True, scene_constraints=scene)
        u = chassis_supplied_opening_speed(point, normal, v, w, .03)
        derivative = u*normal[:2]-np.array([v-w*point[1], w*point[0]])
        self.assertLessEqual((point[:2]-preferred[:2])@derivative, 1.e-8)
        self.assertTrue(np.all(scene[0]@np.array([v, w]) <= scene[1]+1.e-6))
        self.assertLessEqual(abs(v), .06)
        self.assertLessEqual(abs(w), .08)


if __name__ == '__main__':
    unittest.main()
