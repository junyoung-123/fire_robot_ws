import math
import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation
from fire_robot_manipulation.piper_actual_kinematics import (
    PiperActualKinematics, PIPER_BODY_OUTLINE, PIPER_JOINT_LIMITS)


class MinimumContactDisplacementTests(unittest.TestCase):
    def test_nearest_steady_stance_is_not_forced_to_maximum_forward_reach(self):
        arm=PiperActualKinematics()
        observed=np.array([.68,.23,.74])
        rotation=measured_wrist_rotation(arm.approach_rotation,-.25,0.,-.28)
        seed=np.array([0.,1.8,-1.2,0.,-.65,1.5])
        kwargs=dict(observed_curvature=-1.6,minimum_panel_clearance=.08,
                    steady_orientation_only=True)
        old,_=arm.preferred_contact_posture(observed,rotation,seed,.4628,**kwargs)
        new,condition=arm.preferred_contact_posture(observed,rotation,seed,.4628,
                                                  minimum_displacement=True,**kwargs)
        self.assertLess(np.linalg.norm(new[:2]-observed[:2]),np.linalg.norm(old[:2]-observed[:2]))
        self.assertGreaterEqual(condition,.20)
        change=math.asin(-1.6*new[0])-math.atan2(rotation[1,2],rotation[0,2])
        steady=measured_wrist_rotation(rotation,change,0.)
        self.assertGreaterEqual(min((new[:2]-PIPER_BODY_OUTLINE)@steady[:2,2]),.08)
        solution=arm.solve(new,seed=seed,target_rotation=steady,
                           position_tolerance_m=.003,prefer_seed_solution=True)
        self.assertIsNotNone(solution)
        self.assertTrue(np.all(solution.positions>=PIPER_JOINT_LIMITS[:,0]+.20-1.e-9))
        self.assertTrue(np.all(solution.positions<=PIPER_JOINT_LIMITS[:,1]-.20+1.e-9))


if __name__=='__main__':
    unittest.main()
