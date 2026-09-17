import unittest
import numpy as np
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class HoldServoCorrectionTests(unittest.TestCase):
    def node(self):
        node=object.__new__(ManipulationNode)
        node._feedback_config=dict(press_sign=-1.,tracking_limit_m=.04,joint_bias_limit_rad=.18)
        node._feedback_tool=lambda:np.array([.52,.14,.750])
        node._feedback_last_cartesian_target=np.array([.52,.14,.740])
        node._sim_lever_position,node._sim_lever_initial_position=-.13,0.
        node._sim_lever_press_min_angle=.08
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,.02)
        node._feedback_ideal_arm_target=np.zeros(6)
        node._feedback_joint_bias=np.full(6,.03)
        node._feedback_stop=lambda reason:False
        node._feedback_event=lambda *args,**kwargs:None
        node._publish_sim_cmd_vel=lambda v,w:None
        return node

    def test_static_servo_error_is_corrected_without_deeper_cartesian_press(self):
        node=self.node()
        targets=[]
        node._feedback_command_point=lambda point,stage:targets.append(point.copy()) or True
        self.assertTrue(node._feedback_correct_hold_servo())
        np.testing.assert_allclose(targets[0],[.52,.14,.740])
        np.testing.assert_allclose(node._feedback_joint_bias,np.full(6,.02))

    def test_contact_loss_or_excessive_error_cannot_issue_arm_command(self):
        for failure in ('contact','tracking','bias'):
            node=self.node()
            if failure=='contact':node._sim_lever_position=-.01
            if failure=='tracking':node._feedback_tool=lambda:np.array([.52,.14,.8])
            if failure=='bias':node._feedback_joint_bias=np.full(6,-.18)
            node._feedback_command_point=lambda *args:self.fail('Unsafe arm command')
            self.assertFalse(node._feedback_correct_hold_servo())
