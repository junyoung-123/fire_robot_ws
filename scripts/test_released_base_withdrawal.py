"""Exercise measured withdrawal and its refusal cases without a simulator."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from fire_robot_manipulation.observed_angle_manipulation_node import ObservedAngleManipulation
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES,PiperActualKinematics,PIPER_JOINT_LIMITS
from fire_robot_manipulation.released_retreat import plan_released_retreat


def fixture(open_fingers=True, moving=True, stale=False):
    state=NS(now=1.,x=0.,commands=[],events=[],failure=None)
    node=NS(_feedback_config={'tool_tolerance_m':.015},
        _sim_gripper_release_half_width=.034,_feedback_joint_bias=np.zeros(6),
        _piper_kinematics=PiperActualKinematics())
    joints=[.4878106468,2.488959916,-2.120945407,1.393055535,-1.134256665,.055131385]
    state.tool=node._piper_kinematics.forward(joints)[0]
    node._sim_arm_joint_positions=dict(zip(PIPER_JOINT_NAMES,joints))
    node._sim_arm_joint_positions.update(gripper_left_joint=.034 if open_fingers else .01,
                                          gripper_right_joint=-.034 if open_fingers else -.01)
    node._feedback_tool=lambda:None if stale else state.tool.copy()
    def move(point,stage):
        if moving: state.tool=point.copy()
        return True
    node._feedback_move_tool=move
    node._sim_time_sec=lambda:state.now
    node._publish_sim_arm_targets=lambda targets:None
    node._feedback_event=lambda name,**data:state.events.append(dict(event=name,**data))
    def stop(reason): state.failure=reason; return False
    node._feedback_stop=stop
    def command(v,w):
        state.commands.append((v,w))
    node._publish_sim_cmd_vel=command
    return node,state


class TestReleasedWithdrawal(unittest.TestCase):
    def execute(self,**kwargs):
        node,state=fixture(**kwargs)
        with patch('rclpy.ok',return_value=True),patch('time.sleep'):
            result=ObservedAngleManipulation._feedback_retract_released_handle(node,.05)
        return result,state

    def test_completion_uses_measured_displacement(self):
        result,state=self.execute()
        self.assertTrue(result)
        proof=next(e for e in state.events if e['event']=='released_handle_clearance_verified')
        self.assertAlmostEqual(proof['measured_clearance_m'],proof['requested_clearance_m'])
        self.assertEqual(state.commands[-1],(0.,0.))

    def test_closed_gripper_cannot_start_reverse(self):
        result,state=self.execute(open_fingers=False)
        self.assertFalse(result)
        self.assertFalse(any(v<0 for v,w in state.commands))

    def test_unmoved_hand_cannot_complete_by_planned_displacement(self):
        result,state=self.execute(moving=False)
        self.assertFalse(result)
        self.assertEqual(state.commands[-1],(0.,0.))

    def test_stale_feedback_cannot_start_motion(self):
        result,state=self.execute(stale=True)
        self.assertFalse(result)
        self.assertFalse(any(v<0 for v,w in state.commands))

    def test_oversized_tool_cannot_bypass_retreat_bound(self):
        node,state=fixture()
        node._piper_kinematics.tool_front_extent_m=.50
        with patch('rclpy.ok',return_value=True),patch('time.sleep'):
            result=ObservedAngleManipulation._feedback_retract_released_handle(node,.05)
        self.assertFalse(result)
        self.assertEqual(state.commands[-1],(0.,0.))


class TestTFWait(unittest.TestCase):
    def node(self):
        node=object.__new__(ObservedAngleManipulation)
        node.commands=[]
        node._publish_sim_cmd_vel=lambda v,w:node.commands.append((v,w))
        return node

    def test_transient_tf_delay_only_commands_stop(self):
        node=self.node()
        with patch.object(ManipulationNode,'_feedback_base_yaw',
                          side_effect=[ValueError('stale'),.25]),patch('time.sleep'):
            self.assertEqual(node._feedback_base_yaw(),.25)
        self.assertEqual(node.commands,[(0.,0.)])

    def test_persistent_tf_delay_fails_within_bound(self):
        node=self.node()
        with patch.object(ManipulationNode,'_feedback_base_yaw',side_effect=ValueError('stale')), \
                patch('time.monotonic',side_effect=[0.,.6]):
            with self.assertRaises(ValueError): node._feedback_base_yaw()
        self.assertEqual(node.commands,[(0.,0.)])

    def test_opening_exception_becomes_failure_not_node_crash(self):
        node=self.node()
        node._feedback_stop=lambda reason:False
        with patch.object(ManipulationNode,'_feedback_handle_held_open',side_effect=ValueError('stale')):
            self.assertFalse(node._feedback_handle_held_open(2.05))


class TestRetreatPlanner(unittest.TestCase):
    def test_both_wrist_yaw_directions_respect_limits_and_retreat(self):
        model=PiperActualKinematics()
        for sign in (-1.,1.):
            q=np.array([sign*.4878106468,2.488959916,-2.120945407,
                        sign*1.393055535,-1.134256665,sign*.055131385])
            origin,normal,path=plan_released_retreat(model,q,.08)
            previous=0.
            for point,rotation,joints in path:
                distance=float((origin-point)@normal)
                self.assertGreater(distance,previous)
                self.assertTrue(np.all(joints>=PIPER_JOINT_LIMITS[:,0]+.025))
                self.assertTrue(np.all(joints<=PIPER_JOINT_LIMITS[:,1]-.025))
                self.assertLessEqual(np.max(np.abs(joints-q)),.15)
                previous,q=distance,joints
            self.assertAlmostEqual(previous,.08)

    def test_nonfinite_posture_rejected(self):
        with self.assertRaises(ValueError):
            plan_released_retreat(PiperActualKinematics(),[float('nan')]*6,.08)


if __name__=='__main__': unittest.main()
