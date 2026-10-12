import math
import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import contact_posture_command
from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE


class HeadingBarrierTests(unittest.TestCase):
    def test_steering_reconfiguration_respects_body_clearance_rate(self):
        p = np.array([.505, .145, .73])
        n = np.array([math.cos(-.42), math.sin(-.42), 0.])
        preferred = np.array([.52,-.113,.73])
        v, w, target, gap = contact_posture_command(
            p, n, preferred, PIPER_BODY_OUTLINE, door_yaw_rate=-.02)
        np.testing.assert_allclose(target, preferred)
        index = np.argmin((p[:2]-PIPER_BODY_OUTLINE)@n[:2])
        corner = PIPER_BODY_OUTLINE[index]
        tangent = np.array([-n[1],n[0]])
        rate = (.012-v*n[0]+w*(corner[1]*n[0]-corner[0]*n[1])
                -.02*((p[:2]-corner)@tangent))
        self.assertGreaterEqual(rate, -.25*(gap-.03)-1.e-9)

    def test_barrier_is_symmetric_for_reversed_geometry(self):
        p = np.array([.505, -.145, .73])
        n = np.array([math.cos(.42), math.sin(.42), 0.])
        v, w, target, _ = contact_posture_command(
            p, n, p, PIPER_BODY_OUTLINE, door_yaw_rate=.02)
        original = contact_posture_command(
            [p[0],-p[1],p[2]], [n[0],-n[1],0.],
            [p[0],-p[1],p[2]], PIPER_BODY_OUTLINE, door_yaw_rate=-.02)
        self.assertAlmostEqual(v, original[0])
        self.assertAlmostEqual(w, -original[1])
        np.testing.assert_allclose(target, p)

    def test_bad_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            contact_posture_command([.5,.15,.73], [1,0,0], [.5,.15,.73],
                PIPER_BODY_OUTLINE, door_yaw_rate=float('nan'))


if __name__ == '__main__':
    unittest.main()
