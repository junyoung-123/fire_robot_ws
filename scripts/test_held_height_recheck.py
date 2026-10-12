import unittest
from unittest.mock import Mock

import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class HeldHeightRecheckTests(unittest.TestCase):
    def node(self, actual=(.5956929806,.1192780015,.7514965316)):
        node=object.__new__(RobotObservationManipulation)
        node._phase='HANDLE_HELD_BASE_OPEN'
        node._feedback_full_handle_follow=True
        node._feedback_config=dict(tool_tolerance_m=.015,tracking_limit_m=.04)
        node._feedback_last_cartesian_target=np.array([.5979388204,.1136941014,.7384113116])
        node._feedback_tool=lambda:np.array(actual)
        node._feedback_fresh=lambda:True
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._sim_door_position=0.
        node._feedback_base_yaw=lambda:0.
        node._measured_prepress_height=.767
        node._sim_time_sec=lambda:10.
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        node._feedback_stop=Mock(return_value=False)
        return node

    def test_recorded_post_tangent_height_is_rechecked_before_next_probe(self):
        node=self.node()
        before=node._feedback_last_cartesian_target.copy()
        calls=[]
        def settle():
            calls.append(True)
            if len(calls)==2:
                node._feedback_last_cartesian_target[2]=node._feedback_tool()[2]-.0075
            return True
        node._feedback_settle_held_target_once=settle
        self.assertTrue(node._feedback_settle_held_target())
        self.assertEqual(len(calls),2)
        np.testing.assert_allclose(node._feedback_last_cartesian_target[:2],before[:2])
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)
        self.assertEqual(node._feedback_event.call_args.args[0],'held_height_recheck_before_base')

    def test_already_safe_height_keeps_single_original_settle(self):
        node=self.node(actual=(.598,.114,.744))
        node._feedback_settle_held_target_once=Mock(return_value=True)
        self.assertTrue(node._feedback_settle_held_target())
        node._feedback_settle_held_target_once.assert_called_once()

    def test_existing_excessive_vertical_error_is_not_forgiven(self):
        node=self.node(actual=(.598,.114,.756))
        node._feedback_settle_held_target_once=Mock(return_value=True)
        self.assertFalse(node._feedback_settle_held_target())
        node._feedback_stop.assert_called_with('Held height recheck exceeds original pose bounds')

    def test_normal_error_still_stops(self):
        node=self.node(actual=(.616,.114,.744))
        node._feedback_settle_held_target_once=Mock(return_value=True)
        self.assertFalse(node._feedback_settle_held_target())

    def test_stale_or_missing_post_settle_feedback_stops(self):
        for missing in (False,True):
            node=self.node()
            node._feedback_settle_held_target_once=Mock(return_value=True)
            if missing:
                node._feedback_tool=lambda:None
            else:
                node._feedback_fresh=lambda:False
            self.assertFalse(node._feedback_settle_held_target())

    def test_original_settle_failure_never_retries_or_accepts(self):
        node=self.node()
        node._feedback_settle_held_target_once=Mock(return_value=False)
        self.assertFalse(node._feedback_settle_held_target())
        node._feedback_settle_held_target_once.assert_called_once()

    def test_unavailable_prepress_support_does_not_raise_the_goal(self):
        node=self.node()
        node._measured_prepress_height=.74
        before=node._feedback_last_cartesian_target.copy()
        node._feedback_settle_held_target_once=Mock(return_value=True)
        self.assertTrue(node._feedback_settle_held_target())
        np.testing.assert_array_equal(before,node._feedback_last_cartesian_target)

    def test_nonconverging_rechecks_are_bounded(self):
        node=self.node()
        node._feedback_settle_held_target_once=Mock(return_value=True)
        self.assertFalse(node._feedback_settle_held_target())
        self.assertEqual(node._feedback_settle_held_target_once.call_count,4)

    def test_original_other_phase_dispatch_is_unchanged(self):
        node=self.node()
        node._phase='PRESS_LEVER'
        node._feedback_settle_held_target_once=Mock(return_value=True)
        self.assertTrue(node._feedback_settle_held_target())
        node._feedback_settle_held_target_once.assert_called_once()

    def test_nonfinite_measured_or_target_height_stops(self):
        for target in (False,True):
            node=self.node()
            if target:
                node._feedback_last_cartesian_target[2]=float('nan')
            else:
                node._feedback_tool=lambda:np.array([.598,.114,float('nan')])
            node._feedback_settle_held_target_once=Mock(return_value=True)
            self.assertFalse(node._feedback_settle_held_target())

    def test_expired_outer_budget_cannot_resume_base_after_child_settle(self):
        node=self.node()
        now=[10.]
        node._sim_time_sec=lambda:now[0]
        def settle():
            now[0]=19.
            return True
        node._feedback_settle_held_target_once=settle
        self.assertFalse(node._feedback_settle_held_target())


if __name__=='__main__':
    unittest.main()
