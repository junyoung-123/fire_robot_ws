import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.contact_probe_selector import select_contact_probe
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics


class TestContactProbeSelector(unittest.TestCase):
    def test_recorded_near_limit_posture_uses_robot_ik(self):
        model=PiperActualKinematics()
        seed=np.array([.40,1.46,-.58,1.04,-1.18,.74])
        point,rotation=model.forward(seed)
        normal=rotation[:,2].copy(); normal[2]=0.; normal/=np.linalg.norm(normal)
        speed,yaw,info=select_contact_probe(model,point,normal,rotation,seed,
            np.zeros(6),point,.008,-.02,.012)
        self.assertGreaterEqual(speed,0.)
        self.assertGreater(yaw,-.02)
        self.assertGreaterEqual(info['predicted_joint_margin_rad'],.025)
        self.assertLessEqual(info['predicted_obliquity_rad'],.8)

    def test_invalid_feedback_rejected(self):
        with self.assertRaises(ValueError):
            select_contact_probe(PiperActualKinematics(),[float('nan'),0,0],[1,0,0],
                np.eye(3),np.zeros(6),np.zeros(6),np.zeros(3),.01,0,.012)


if __name__=='__main__': unittest.main()
