from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.minimal_observed_angle_node import MinimalObservedAngle
from fire_robot_manipulation.preserved_contact_policy import ManipulationNode


class PanelStreamRecoveryTests(unittest.TestCase):
    def fixture(self, recover_after=None, update_encoders=True):
        node=object.__new__(RobotObservationManipulation)
        clock={'wall':100.,'sim':10.}
        node._sim_time_sec=lambda:clock['sim']
        node._feedback_config={'max_age_sec':1.}
        node._feedback_door_msg_stamp=(10,0)
        node._feedback_door_stamp=100.
        node._feedback_arm_stamp=100.
        node._lever_tracking=False
        node._feedback_lever_stamp=None
        node._phase='HANDLE_HELD_BASE_OPEN'
        node._feedback_measurement_pause_generation=0
        node._angle_freshness=NS(valid=lambda *args:recover_after is not None and clock['wall']-100.>=recover_after)
        node._publish_sim_cmd_vel=Mock();node._feedback_event=Mock()
        def sleep(_):
            clock['wall']+=.5;clock['sim']+=.1
            if update_encoders:node._feedback_arm_stamp=clock['wall']
        return node,clock,sleep

    def run_wait(self,node,clock,sleep):
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',side_effect=lambda:clock['wall']), \
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.sleep',side_effect=sleep), \
             patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True):
            return node._feedback_fresh()

    def test_slow_panel_reacquisition_recovers_only_with_chassis_stopped(self):
        node,clock,sleep=self.fixture(recover_after=3.)
        self.assertTrue(self.run_wait(node,clock,sleep))
        self.assertGreater(clock['wall']-100.,2.)
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
        self.assertEqual(node._feedback_measurement_pause_generation,1)

    def test_missing_panel_still_has_bounded_simulation_wait(self):
        node,clock,sleep=self.fixture()
        self.assertFalse(self.run_wait(node,clock,sleep))
        self.assertLessEqual(clock['sim']-10.,2.1)
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_missing_own_encoders_does_not_get_extended_panel_wait(self):
        node,clock,sleep=self.fixture(recover_after=10.,update_encoders=False)
        self.assertFalse(self.run_wait(node,clock,sleep))
        self.assertLessEqual(clock['wall']-100.,2.)

    def aligned_lever_fixture(self,recover_after=3.,pending=True,update_encoders=True):
        node,clock,sleep=self.fixture(recover_after=0.,update_encoders=update_encoders)
        node._lever_tracking=True
        node._encoder_panel_agreement=True
        node._lever_estimate_valid=False
        node._lever_alignment_pending=pending
        node._feedback_lever_stamp=100.
        node._feedback_joint_msg_stamp=(10,0)
        def update(duration):
            sleep(duration)
            clock['sim']=round(clock['sim'],9)
            if update_encoders:
                sec=int(clock['sim'])
                node._feedback_joint_msg_stamp=(sec,round((clock['sim']-sec)*1.e9))
            if recover_after is not None and clock['wall']-100.>=recover_after:
                node._lever_estimate_valid=True
                node._lever_alignment_pending=False
                node._feedback_lever_stamp=clock['wall']
        return node,clock,update

    def test_tf_reacquisition_with_fresh_own_encoders_waits_only_while_stopped(self):
        node,clock,sleep=self.aligned_lever_fixture()
        self.assertTrue(self.run_wait(node,clock,sleep))
        self.assertGreater(clock['wall']-100.,2.)
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_tf_pending_does_not_extend_missing_encoders_or_grasp_loss(self):
        for updates in (dict(pending=False),dict(update_encoders=False)):
            with self.subTest(updates=updates):
                node,clock,sleep=self.aligned_lever_fixture(recover_after=10.,**updates)
                self.assertFalse(self.run_wait(node,clock,sleep))
                self.assertLessEqual(clock['wall']-100.,2.)

    def test_tf_reacquisition_retains_existing_simulation_budget(self):
        node,clock,sleep=self.aligned_lever_fixture(recover_after=None)
        self.assertFalse(self.run_wait(node,clock,sleep))
        self.assertLessEqual(clock['sim']-10.,2.1)

    def test_future_latest_encoder_uses_original_fresh_history_for_stopped_recovery(self):
        node,clock,sleep=self.aligned_lever_fixture()
        node._lever_encoder_history=[]
        def update(duration):
            sleep(duration)
            stamp=node._feedback_joint_msg_stamp
            node._lever_encoder_history.append((stamp,clock['wall'],{}))
            future=clock['sim']+.001
            node._feedback_joint_msg_stamp=(int(future),round((future-int(future))*1.e9))
        self.assertTrue(self.run_wait(node,clock,update))
        self.assertGreater(clock['wall']-100.,2.)
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_stale_history_does_not_extend_stopped_alignment_wait(self):
        node,clock,sleep=self.aligned_lever_fixture(recover_after=10.)
        node._lever_encoder_history=[((9,0),99.,{})]
        def update(duration):
            sleep(duration)
            future=clock['sim']+.001
            node._feedback_joint_msg_stamp=(int(future),round((future-int(future))*1.e9))
        self.assertFalse(self.run_wait(node,clock,update))
        self.assertLessEqual(clock['wall']-100.,2.)

    def test_atomic_pending_snapshot_cannot_be_mixed_with_callback_flags(self):
        node,clock,sleep=self.aligned_lever_fixture()
        node._lever_alignment_pending=False
        node._lever_observation_state=(False,True,None,None,'TF pending')
        def update(duration):
            sleep(duration)
            if node._lever_estimate_valid:
                node._lever_observation_state=(True,False,clock['wall'],node._feedback_joint_msg_stamp,None)
        self.assertTrue(self.run_wait(node,clock,update))
        self.assertGreater(clock['wall']-100.,2.)
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_time_is_sampled_after_encoder_receipt_snapshot(self):
        node,clock,sleep=self.aligned_lever_fixture(recover_after=0.)
        node._lever_estimate_valid=True
        node._lever_observation_state=(True,False,100.,(10,0),None)
        calls=[0]
        def clock_with_concurrent_callback():
            calls[0]+=1
            if calls[0]>1:
                node._feedback_arm_stamp=clock['wall']+.001
                node._lever_observation_state=(True,False,clock['wall']+.001,(10,0),None)
            return clock['wall']
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',
                   side_effect=clock_with_concurrent_callback), \
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.sleep',side_effect=sleep), \
             patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True):
            # The first clock read establishes the wait start; subsequent reads
            # can trigger an arrival after a coherent input snapshot.
            node._feedback_arm_stamp=100.
            self.assertTrue(node._feedback_fresh())
        node._publish_sim_cmd_vel.assert_not_called()

    def test_adapter_does_not_subscribe_twice_to_one_observed_angle(self):
        node=object.__new__(MinimalObservedAngle)
        node.get_parameter=Mock(return_value=NS(value=True))
        node.create_subscription=Mock()
        def base_init(this):
            this._sim_door_feedback_joint_name='camera_door_rotation'
            this._sim_door_feedback_topic='/observed_door/rotation'
            this._sim_door_state_subscription=object()
        with patch.object(ManipulationNode,'__init__',new=base_init):
            MinimalObservedAngle.__init__(node)
        node.create_subscription.assert_not_called()


if __name__=='__main__':
    unittest.main()
