import unittest
from unittest.mock import patch
import numpy as np
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_LIMITS


class ContactIkFastPathTests(unittest.TestCase):
    def test_contact_posture_uses_observed_height_and_robot_bounds(self):
        model = PiperActualKinematics()
        observed = np.array([.607,.15,.733])
        rotation = measured_wrist_rotation(model.approach_rotation,-.25,0.,-.24)
        seed = np.array([.08,1.82,-1.07,.46,-.83,.99])
        preferred,ratio = model.preferred_contact_posture(observed,rotation,seed,.463)
        self.assertGreater(ratio,.35)
        self.assertGreaterEqual(preferred[0],.463)
        self.assertLessEqual(preferred[0],observed[0])
        np.testing.assert_allclose(preferred[1:],observed[1:])

    def test_valid_current_pose_does_not_search_global_seeds(self):
        model = PiperActualKinematics()
        joints = np.array([.1,1.8,-1.1,.4,-.8,1.0])
        point, rotation = model.forward(joints)
        with patch.object(model,'_error',wraps=model._error) as error:
            result = model.solve(point, seed=joints, target_rotation=rotation,
                prefer_seed_solution=True, position_tolerance_m=.001, orientation_tolerance_rad=.03)
        self.assertEqual(error.call_count,1)
        np.testing.assert_allclose(result.positions,joints)

    def test_small_updates_still_require_position_rotation_and_joint_limits(self):
        model = PiperActualKinematics()
        joints = np.array([.1,1.8,-1.1,.4,-.8,1.0])
        for offset in (-.02,.02,.04):
            moved = joints + np.array([offset,0.,0.,0.,0.,0.])
            point, rotation = model.forward(moved)
            result = model.solve(point, seed=joints, target_rotation=rotation,
                prefer_seed_solution=True, position_tolerance_m=.001, orientation_tolerance_rad=.03)
            self.assertIsNotNone(result)
            self.assertLessEqual(result.position_error_m,.001)
            self.assertLessEqual(result.orientation_error_rad,.03)
            self.assertTrue(np.all(result.positions >= PIPER_JOINT_LIMITS[:,0]))
            self.assertTrue(np.all(result.positions <= PIPER_JOINT_LIMITS[:,1]))
