"""Regression pose recorded from R28; never a runtime door target."""
import unittest
import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_LIMITS


class IkBoundaryTests(unittest.TestCase):
    def test_observed_approach_boundary_converges_without_looser_tolerance(self):
        kin=PiperActualKinematics()
        seed=[-.051466,.358489,-.51006,.256145,.282466,1.338867]
        target=[.3581889318156076,.1576248714792187,.8492066624431261]
        result=kin.solve(target,seed=seed,position_tolerance_m=.003)
        self.assertIsNotNone(result)
        self.assertLessEqual(result.position_error_m,.003)
        self.assertLessEqual(result.orientation_error_rad,.10)
        self.assertTrue(np.all(result.positions>=PIPER_JOINT_LIMITS[:,0]))
        self.assertTrue(np.all(result.positions<=PIPER_JOINT_LIMITS[:,1]))


if __name__=='__main__': unittest.main()
