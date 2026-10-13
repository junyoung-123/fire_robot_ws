"""Only a confirmed press command may shift the subsequent contact reference."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from scripts import test_loaded_settle_reserve as fixtures


class ConfirmedHoldReferenceTests(unittest.TestCase):
    def node(self):
        node=fixtures.LoadedSettleReserveTests().node()
        node._feedback_follow_mode='encoder_lever_contact'
        node._feedback_last_cartesian_target=np.array([.5220805506,.1017181645,.7570894528])
        node._feedback_sweep_state=(np.array([5.,2.,.757]),np.array([5.01,2.,.765]),.38)
        node._feedback_press_origin=np.array([.5220805506,.1017181645,.7647738755])
        node._feedback_press_depth=.0076844227
        node._feedback_tool=lambda: np.array([.5061567918,.1005890490,.7628155229])
        node._feedback_base_yaw=lambda: 0.
        node._sim_door_position=0.
        node._sim_door_initial_position=0.
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._tf_buffer.lookup_transform.return_value=SimpleNamespace(
            transform=SimpleNamespace(rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)))
        node._feedback_config.update(lever_hold_tolerance_rad=.02,press_step_m=.005)
        node._feedback_lever_hold_delta=.18
        node._sim_lever_position=-.13
        node._feedback_press_increment=lambda *args:.00375
        node._feedback_confirm_latch_press_response=Mock(return_value=True)
        return node

    @staticmethod
    def corrected():
        return np.array([.5122504769,.0994334078,.7533394528])

    def test_recorded_v7_command_shift_is_retained(self):
        node=self.node()
        old=node._feedback_last_cartesian_target.copy()
        reference,previous,angle=node._feedback_sweep_state
        self.assertTrue(node._feedback_confirmed_hold_reference(old,self.corrected()))
        delta=self.corrected()-old
        delta[2]=0.
        np.testing.assert_allclose(node._feedback_sweep_state[0],reference+delta)
        np.testing.assert_array_equal(node._feedback_sweep_state[1],previous)
        self.assertEqual(node._feedback_sweep_state[2],angle)
        np.testing.assert_allclose(node._feedback_press_origin[:2],self.corrected()[:2])
        node._feedback_stop.assert_not_called()

    def test_shift_rotates_into_own_reference_frame(self):
        node=self.node()
        q=node._tf_buffer.lookup_transform.return_value.transform.rotation
        q.z=q.w=math.sqrt(.5)
        reference=node._feedback_sweep_state[0].copy()
        old=node._feedback_last_cartesian_target.copy()
        self.assertTrue(node._feedback_confirmed_hold_reference(old,self.corrected()))
        delta=self.corrected()-old
        np.testing.assert_allclose(node._feedback_sweep_state[0]-reference,[-delta[1],delta[0],0.],atol=1.e-12)

    def dispatch_node(self):
        node=self.node()
        node._feedback_tool=Mock(side_effect=[
            np.array([.5122504769,.0994334078,.7645894528]),
            np.array([.5061567918,.1005890490,.7628155229])])
        return node

    def test_failed_response_cannot_change_reference(self):
        node=self.dispatch_node()
        reference=node._feedback_sweep_state[0].copy()
        node._feedback_confirm_latch_press_response.return_value=False
        self.assertFalse(ManipulationNode._feedback_retain_lever_press(node))
        np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)
        node._tf_buffer.lookup_transform.assert_not_called()

    def test_large_shift_or_invalid_feedback_cannot_rebase(self):
        for issue in ('shift','stale','normal','height','nan'):
            node=self.node()
            reference=node._feedback_sweep_state[0].copy()
            old=node._feedback_last_cartesian_target.copy()
            goal=self.corrected()
            if issue=='shift':
                goal[0]-=.02
            elif issue=='stale':
                node._feedback_fresh=lambda:False
            else:
                delta={'normal':[.016,0.,0.],'height':[0.,0.,.016],'nan':[float('nan'),0.,0.]}[issue]
                node._feedback_tool=lambda:goal+delta
            self.assertFalse(node._feedback_confirmed_hold_reference(old,goal))
            np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)
            node._feedback_stop.assert_called_once()

    def test_transform_failure_stops_without_rebasing(self):
        node=self.node()
        reference=node._feedback_sweep_state[0].copy()
        node._tf_buffer.lookup_transform.side_effect=ValueError('unavailable')
        self.assertFalse(node._feedback_confirmed_hold_reference(node._feedback_last_cartesian_target,self.corrected()))
        np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)

    def test_other_modes_are_unchanged(self):
        for attribute,value in (('_feedback_follow_mode','vision'),('_feedback_full_handle_follow',False),('_phase','PRE_GRASP')):
            node=self.node()
            setattr(node,attribute,value)
            reference=node._feedback_sweep_state[0].copy()
            self.assertTrue(node._feedback_confirmed_hold_reference(node._feedback_last_cartesian_target,self.corrected()))
            np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)

    def test_hook_runs_once_after_actual_retention_response(self):
        node=self.dispatch_node()
        old=node._feedback_last_cartesian_target.copy()
        reference=node._feedback_sweep_state[0].copy()
        self.assertTrue(ManipulationNode._feedback_retain_lever_press(node))
        node._feedback_confirm_latch_press_response.assert_called_once()
        delta=self.corrected()-old
        delta[2]=0.
        np.testing.assert_allclose(node._feedback_sweep_state[0],reference+delta)
        events=[c for c in node._feedback_event.call_args_list if c.args[0]=='confirmed_hold_reference_shift']
        self.assertEqual(len(events),1)

    def test_r34_settling_without_new_press_does_not_rebase_twice(self):
        node=self.node()
        node._contact_motion_policy='r34_reference'
        node._sim_lever_position=-.18
        reference=node._feedback_sweep_state[0].copy()
        delta=np.array([.006,.002,0.])
        def settle():
            node._feedback_sweep_state=(reference+delta,np.zeros(3),.38)
            node._feedback_last_cartesian_target+=delta+[0.,0.,.001]
            node._feedback_press_depth-=.001
            return True
        node._feedback_settle_held_target=settle
        self.assertTrue(node._feedback_retain_lever_press())
        np.testing.assert_allclose(node._feedback_sweep_state[0],reference+delta)
        node._feedback_confirm_latch_press_response.assert_not_called()
        node._tf_buffer.lookup_transform.assert_not_called()

    def test_r34_settling_then_press_uses_post_settling_reference(self):
        node=self.dispatch_node()
        node._contact_motion_policy='r34_reference'
        reference=node._feedback_sweep_state[0].copy()
        old=node._feedback_last_cartesian_target.copy()
        delta=np.array([-.003,.002,0.])
        def settle():
            node._feedback_sweep_state=(reference+delta,np.zeros(3),.38)
            node._feedback_last_cartesian_target+=delta+[0.,0.,.001]
            node._feedback_press_depth-=.001
            return True
        node._feedback_settle_held_target=settle
        self.assertTrue(node._feedback_retain_lever_press())
        expected=self.corrected()-old
        expected[2]=0.
        np.testing.assert_allclose(node._feedback_sweep_state[0],reference+expected)

    def test_default_hook_is_noop(self):
        self.assertTrue(ManipulationNode._feedback_confirmed_hold_reference(None,None,None))


if __name__=='__main__':
    unittest.main()
