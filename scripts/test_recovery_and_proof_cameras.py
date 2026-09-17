import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_NAMES, PIPER_STOW

path = Path(__file__).resolve().parents[1]/'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py'
spec = importlib.util.spec_from_file_location('proof_camera_test_module',path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ProofCameraTests(TestCase):
    def test_door_or_robot_pose_cannot_be_commanded(self):
        for name in ('fire_robot','blue_hinged_contact_door','proof_handle_camera" evil'):
            with self.assertRaises(ValueError):
                module.proof_camera_request(name,[0,0,0],[1,0,0])

    def test_camera_x_axis_points_at_target(self):
        for target in ([1,0,0],[1,2,3],[-1,2,-1]):
            request = module.proof_camera_request('proof_handle_camera',[0,0,0],target)
            q = [float(v) for v in re.findall(r'[xyzw]: ([^ }]+)',request.split('orientation')[1])]
            x,y,z,w = q
            forward = np.array([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)])
            np.testing.assert_allclose(forward,np.array(target)/np.linalg.norm(target),atol=1.e-12)

    def test_invalid_camera_geometry_rejected(self):
        for target in ([0,0,0],[1,float('nan'),0],[1,2]):
            with self.assertRaises(ValueError):
                module.proof_camera_request('proof_handle_camera',[0,0,0],target)

    def test_recovery_panel_collision_cannot_pass(self):
        sample = dict(part='panel',phase='RETRACT_FROM_HANDLE',contacts=[
            dict(collision1='door::panel',collision2='fire_robot::gripper_finger_right')])
        self.assertFalse(module.audit_recovery_contacts([sample])['pass'])
        sample['part']='lever'
        self.assertTrue(module.audit_recovery_contacts([sample])['pass'])

    def test_empty_contact_heartbeat_is_valid_but_absent_stream_is_not(self):
        samples=[dict(part=part,sim_time_sec=t,contacts=[]) for t in (1.,1.1,1.2)
                 for part in ('lever','latch','panel','jamb','strike')]
        self.assertTrue(module.audit_contact_coverage(samples)['pass'])
        self.assertFalse(module.audit_contact_coverage([s for s in samples if s['part']!='panel'])['pass'])

    def test_contact_heartbeat_gap_cannot_pass(self):
        samples=[dict(part=part,sim_time_sec=t,contacts=[]) for t in (1.,1.1,2.)
                 for part in ('lever','latch','panel','jamb','strike')]
        self.assertFalse(module.audit_contact_coverage(samples)['pass'])


class HeldRecoveryTests(TestCase):
    def test_recovery_lifts_instead_of_sweeping_back_into_panel(self):
        node = object.__new__(ManipulationNode)
        q = np.array([-.09,1.02,-.22,.36,-.82,1.06])
        node._sim_arm_joint_positions = dict(zip(PIPER_JOINT_NAMES,q))
        node._sim_last_arm_target = q.copy()
        node._feedback_contact = node._hold_handle_during_base_open = True
        node.get_logger = lambda: SimpleNamespace(error=lambda *args:None)
        commands=[]
        node._command_sim_arm_targets=lambda stage,values: commands.append(
            (stage,np.array([values[n] for n in PIPER_JOINT_NAMES]))) or True
        node._wait_for_sim_arm_target=lambda **kwargs:True
        self.assertTrue(node._move_sim_arm_to_stow())
        k=PiperActualKinematics()
        start,rotation=k.forward(q)
        for _,next_q in commands:
            for t in np.linspace(0,1,101):
                point,_=k.forward(q*(1-t)+next_q*t)
                self.assertGreaterEqual(point[2],start[2]-.001)
                self.assertLessEqual(float((point-start)@rotation[:,2]),.001)
            q=next_q
        np.testing.assert_allclose(q,PIPER_STOW)
