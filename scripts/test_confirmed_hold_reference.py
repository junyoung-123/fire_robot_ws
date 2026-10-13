"""A response-confirmed press must not be undone by the next following call."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from scripts import test_loaded_settle_reserve as fixtures


class ConfirmedHoldReferenceTests(unittest.TestCase):
    def node(self):
        node=fixtures.LoadedSettleReserveTests().node()
        node._feedback_follow_mode='encoder_lever_contact'
        node._feedback_last_cartesian_target=np.array([.5220805506,.1017181645,.7570894528])
        node._feedback_sweep_state=(np.array([5.,2.,.757]),np.array([5.01,2.,.765]),.38)
        node._feedback_press_depth=.0076844227
        node._feedback_tool=lambda: np.array([.5061567918,.1005890490,.7628155229])
        node._feedback_base_yaw=lambda: 0.
        node._sim_door_position=0.
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._tf_buffer.lookup_transform.return_value=SimpleNamespace(
            transform=SimpleNamespace(rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)))
        return node

    def correct(self,node,result=True):
        node._feedback_last_cartesian_target=np.array([.5122504769,.0994334078,.7533394528])
        node._feedback_press_depth=.0114344227
        return result

    def test_recorded_v7_press_shift_is_retained_without_passive_drift(self):
        node=self.node()
        old_goal=node._feedback_last_cartesian_target.copy()
        reference,previous,angle=node._feedback_sweep_state
        with patch.object(ManipulationNode,'_feedback_retain_lever_press',side_effect=lambda:self.correct(node)):
            self.assertTrue(node._feedback_retain_lever_press())
        shift=node._feedback_last_cartesian_target-old_goal
        shift[2]=0.
        np.testing.assert_allclose(node._feedback_sweep_state[0],reference+shift)
        np.testing.assert_array_equal(node._feedback_sweep_state[1],previous)
        self.assertEqual(node._feedback_sweep_state[2],angle)
        np.testing.assert_allclose(node._feedback_press_origin[:2],node._feedback_last_cartesian_target[:2])
        node._feedback_stop.assert_not_called()
        # Without another active, confirmed press the reference must not move.
        retained=node._feedback_sweep_state[0].copy()
        with patch.object(ManipulationNode,'_feedback_retain_lever_press',return_value=True):
            for _ in range(3):
                self.assertTrue(node._feedback_retain_lever_press())
        np.testing.assert_array_equal(node._feedback_sweep_state[0],retained)

    def test_shift_rotates_into_own_reference_frame(self):
        node=self.node()
        q=node._tf_buffer.lookup_transform.return_value.transform.rotation
        q.z=q.w=math.sqrt(.5)
        reference=node._feedback_sweep_state[0].copy()
        old=node._feedback_last_cartesian_target.copy()
        with patch.object(ManipulationNode,'_feedback_retain_lever_press',side_effect=lambda:self.correct(node)):
            self.assertTrue(node._feedback_retain_lever_press())
        delta=node._feedback_last_cartesian_target-old
        np.testing.assert_allclose(node._feedback_sweep_state[0]-reference,[-delta[1],delta[0],0.],atol=1.e-12)

    def test_failed_response_cannot_change_reference(self):
        node=self.node()
        reference=node._feedback_sweep_state[0].copy()
        with patch.object(ManipulationNode,'_feedback_retain_lever_press',side_effect=lambda:self.correct(node,False)):
            self.assertFalse(node._feedback_retain_lever_press())
        np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)
        node._tf_buffer.lookup_transform.assert_not_called()

    def test_large_shift_or_invalid_feedback_cannot_rebase(self):
        for issue in ('shift','stale','normal','height','nan'):
            node=self.node()
            reference=node._feedback_sweep_state[0].copy()
            def correction():
                self.correct(node)
                if issue=='shift':
                    node._feedback_last_cartesian_target[0]-=.02
                elif issue=='stale':
                    node._feedback_fresh=lambda:False
                else:
                    delta={'normal':[.016,0.,0.],'height':[0.,0.,.016],'nan':[float('nan'),0.,0.]}[issue]
                    node._feedback_tool=lambda:node._feedback_last_cartesian_target+delta
                return True
            with patch.object(ManipulationNode,'_feedback_retain_lever_press',side_effect=correction):
                self.assertFalse(node._feedback_retain_lever_press())
            np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)
            node._feedback_stop.assert_called_once()

    def test_transform_failure_stops_without_rebasing(self):
        node=self.node()
        reference=node._feedback_sweep_state[0].copy()
        node._tf_buffer.lookup_transform.side_effect=ValueError('unavailable')
        with patch.object(ManipulationNode,'_feedback_retain_lever_press',side_effect=lambda:self.correct(node)):
            self.assertFalse(node._feedback_retain_lever_press())
        np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)
        node._feedback_stop.assert_called_once()

    def test_other_modes_are_unchanged(self):
        for attribute,value in (('_feedback_follow_mode','vision'),('_feedback_full_handle_follow',False),('_phase','PRE_GRASP')):
            node=self.node()
            setattr(node,attribute,value)
            reference=node._feedback_sweep_state[0].copy()
            with patch.object(ManipulationNode,'_feedback_retain_lever_press',return_value=True) as original:
                self.assertTrue(node._feedback_retain_lever_press())
            original.assert_called_once()
            np.testing.assert_array_equal(node._feedback_sweep_state[0],reference)


if __name__=='__main__':
    unittest.main()
