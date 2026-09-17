import unittest
from types import SimpleNamespace as NS
import numpy as np
from fire_robot_manipulation.contact_feedback import released_grip_clearance
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class ReleasedGripTests(unittest.TestCase):
    def test_clearance_comes_from_tool_geometry_and_measured_span(self):
        self.assertAlmostEqual(released_grip_clearance(.1358,.1,.052,.015),.0768)
        self.assertGreater(released_grip_clearance(.1358,.1,.06,.015),
                           released_grip_clearance(.1358,.1,.04,.015))

    def test_invalid_grasp_span_cannot_select_clearance(self):
        for span in (0.,float('nan'),.2):
            with self.assertRaises(ValueError):
                released_grip_clearance(.1358,.1,span,.015)

    def test_retract_uses_small_steps_and_freezes_measured_wrist(self):
        node=object.__new__(ManipulationNode)
        pose=[np.array([.6,.15,.75])]
        rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        node._feedback_tool=lambda:pose[0].copy()
        node._publish_sim_cmd_vel=lambda *args:None
        node._feedback_stop=lambda reason:False
        node._feedback_event=lambda *args,**kwargs:None
        node._feedback_config=dict(tool_tolerance_m=.015,approach_step_m=.02)
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._piper_kinematics=NS(forward=lambda q:(pose[0],rotation),
            tool_front_extent_m=.1358,tool_contact_offset_m=.1)
        steps=[]
        def move(goal,stage):
            steps.append(np.linalg.norm(goal-pose[0]))
            pose[0]=goal.copy()
            return True
        node._feedback_move_tool=move
        self.assertTrue(node._feedback_retract_released_handle(.052))
        self.assertLessEqual(max(steps),.02)
        self.assertAlmostEqual(pose[0][0],.6-.0768)
        np.testing.assert_allclose(node._feedback_contact_rotation(),rotation)
