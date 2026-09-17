import math
import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import coordinated_contact_endpoint, contact_posture_command
from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE


class CoordinatedContactTests(unittest.TestCase):
    def test_normal_forward_motion_keeps_arm_fixed(self):
        start = np.array([.65,.15,.75])
        end, _ = coordinated_contact_endpoint(start, [1.,0.,0.], .02, 0., .3, .008)
        np.testing.assert_allclose(start, end)

    def test_rotating_base_cancels_perpendicular_drag(self):
        start = np.array([.65,.15,.75])
        normal = np.array([math.cos(-.6),math.sin(-.6),0.])
        v, w, duration = .02, -.04, .15
        end, inv = coordinated_contact_endpoint(start, normal, v, w, duration, .008)
        translation = np.array([v/w*math.sin(w*duration), v/w*(1.-math.cos(w*duration)),0.])
        delta_world = inv.T@end + translation-start
        self.assertGreater(delta_world@normal, 0.)
        self.assertAlmostEqual(delta_world@np.array([-normal[1],normal[0],0.]), 0.)
        self.assertAlmostEqual(end[2], start[2])

    def test_rejects_excessive_step_or_closing_motion(self):
        for normal, duration in (([1.,0.,0.],2.), ([-1.,0.,0.],.1)):
            with self.assertRaises(ValueError):
                coordinated_contact_endpoint([.65,.15,.75], normal, .02, 0., duration, .008)

    def test_rejects_nonfinite_feedback(self):
        with self.assertRaises(ValueError):
            coordinated_contact_endpoint([float('nan'),.15,.75], [1.,0.,0.], .02, 0., .1, .008)

    def test_base_can_catch_up_without_accelerating_the_handle(self):
        start = np.array([.65,.15,.75])
        end, _ = coordinated_contact_endpoint(start, [1.,0.,0.], .04, 0., .1, .008, .01)
        self.assertAlmostEqual(end[0], .647)

    def test_camera_clear_posture_is_derived_from_observed_normal(self):
        normal = np.array([math.sqrt(1.-.85**2),-.85,0.])
        _, _, target, _ = contact_posture_command([.54,.15,.75], normal,
            [.54,.15,.75], PIPER_BODY_OUTLINE)
        self.assertLess(target[1], -.08)
        self.assertAlmostEqual(target[:2]@normal[:2]-max(PIPER_BODY_OUTLINE@normal[:2]), .03)

    def test_reach_error_increases_base_speed_not_opening_speed(self):
        preferred = np.array([.54,.15,.75])
        nominal = contact_posture_command(preferred,[1.,0.,0.],preferred,PIPER_BODY_OUTLINE)
        extended = contact_posture_command([.70,.15,.75],[1.,0.,0.],preferred,PIPER_BODY_OUTLINE)
        self.assertGreater(extended[0], nominal[0])
        self.assertLessEqual(extended[0], .06)

    def test_posture_controller_satisfies_local_velocity_equations(self):
        point = np.array([.57,.10,.75])
        normal = np.array([math.cos(-.5),math.sin(-.5),0.])
        v,w,target,_ = contact_posture_command(point, normal,[.54,.15,.75],PIPER_BODY_OUTLINE)
        relative = .012*normal-np.array([v-w*point[1],w*point[0],0.])
        np.testing.assert_allclose(relative[:2], .25*(target-point)[:2])
