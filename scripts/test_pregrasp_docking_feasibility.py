import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics
from fire_robot_manipulation.mission_contact_adapter import MissionContactAdapter


class PregraspDockingTests(unittest.TestCase):
    def test_recorded_close_grasp_is_not_sufficient_for_reachable_pregrasp(self):
        arm=PiperActualKinematics()
        observed=np.array([.474316716,.114790050,.768771378])
        approach=observed.copy();approach[0]-=.073375448
        self.assertIsNone(arm.solve(approach,max_iterations=100,position_tolerance_m=.003))
        minimum=.312+arm.tool_front_extent_m+.015
        posture=arm.docking_contact_posture([.8,observed[1],observed[2]],minimum,
            candidate_lanes=[observed[1]],prefer_compact=True,
            pregrasp_clearance_m=.073375448)
        approach=posture.copy();approach[0]-=.073375448
        self.assertIsNotNone(arm.solve(posture,minimum_joint_margin_rad=.20))
        self.assertIsNotNone(arm.solve(approach,position_tolerance_m=.003))
        self.assertGreater(posture[0],observed[0])

    def test_unbounded_or_invalid_pregrasp_clearance_is_rejected(self):
        arm=PiperActualKinematics()
        for clearance in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                arm.docking_contact_posture([.8,.15,.77],.4628,pregrasp_clearance_m=clearance)

    def adapter(self):
        node=MissionContactAdapter.__new__(MissionContactAdapter)
        node.now=Mock(return_value=20.2)
        node.docking_posture=np.array([.48,.12,.77]);node.grip_lane_y=.12
        node.handle_positions=deque([(20.,np.array([5.,-1.,.77])),
            (20.05,np.array([5.001,-1.,.77])),(20.1,np.array([4.999,-1.,.77]))])
        node.handle_spans=deque([.1,.1,.1])
        node.camera_xyz=np.array([.25,.0175,.789]);node.camera_bearing_limit=.6
        node.kinematics=NS(tool_front_extent_m=.1358,tool_contact_offset_m=.1,
            contact_reach_upper_bound=Mock(return_value=.85),
            docking_contact_posture=Mock(return_value=np.array([.51,.12,.77])),
            solve=Mock(side_effect=[NS(positions=np.zeros(6)),NS(positions=np.zeros(6))]))
        return node

    def test_live_observation_spread_controls_both_endpoint_checks(self):
        node=self.adapter();xyz=np.array([.51,.12,.77])
        solution,posture=node.alignment_solution(xyz)
        self.assertIsNotNone(solution)
        options=node.kinematics.docking_contact_posture.call_args.kwargs
        self.assertAlmostEqual(options['pregrasp_clearance_m'],.0728)
        np.testing.assert_allclose(node.kinematics.solve.call_args_list[0].args[0],xyz-[.0728,0.,0.])
        np.testing.assert_allclose(posture,[.51,.12,.77])

    def test_stale_observations_never_select_a_new_docking_pose(self):
        node=self.adapter();node.now.return_value=21.
        solution,_=node.alignment_solution(np.array([.5,.12,.77]))
        self.assertIsNone(solution)
        node.kinematics.docking_contact_posture.assert_not_called()

    def test_unreachable_current_pregrasp_blocks_the_grasp_test(self):
        node=self.adapter();node.kinematics.solve.side_effect=[None]
        solution,posture=node.alignment_solution(np.array([.47,.12,.77]))
        self.assertIsNone(solution)
        self.assertEqual(node.kinematics.solve.call_count,1)
        np.testing.assert_allclose(posture,[.51,.12,.77])


if __name__=='__main__':
    unittest.main()
