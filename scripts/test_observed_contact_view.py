import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.observed_contact_view import view_preserving_probe, joint_limited_view_obliquity


class TestViewProbe(unittest.TestCase):
    outline=np.array([[-.25,-.23],[-.25,.23],[.25,-.23],[.25,.23]])

    def test_small_obliquity_keeps_original_command(self):
        result=view_preserving_probe([.55,.15,.75],[1.,0.,0.],self.outline,.02,-.01,.012)
        self.assertEqual(result,(.02,-.01,None))

    def test_correction_turns_toward_observed_panel(self):
        a=-.9
        normal=np.array([math.cos(a),math.sin(a),0.])
        speed,yaw,diagnostic=view_preserving_probe([.55,-.10,.75],normal,self.outline,.02,-.01,.012)
        self.assertIsNotNone(diagnostic)
        self.assertLess(yaw,-.01)
        self.assertGreaterEqual(speed,0.)
        self.assertLessEqual(abs(yaw),.08)
        self.assertGreaterEqual(diagnostic['predicted_clearance_m'],
                                min(diagnostic['current_clearance_m'],.035))

    def test_no_motion_from_invalid_observation(self):
        with self.assertRaises(ValueError):
            view_preserving_probe([.5,0,.7],[float('nan'),0.,0.],self.outline,.02,0.,.012)

    def test_joint_margin_tightens_heading_before_limit(self):
        limits=np.array([[-1.,1.]]*6)
        healthy,margin=joint_limited_view_obliquity(np.zeros(6),limits,np.zeros(6))
        near,remaining=joint_limited_view_obliquity([0,0,0,0,-.85,0],limits,np.ones(6)*.01)
        self.assertEqual(healthy,.5)
        self.assertLess(near,healthy)
        self.assertAlmostEqual(remaining,.14)
        np.testing.assert_array_equal(limits,np.array([[-1.,1.]]*6))


if __name__=='__main__': unittest.main()
