"""Verify that simulator hinge values cannot overwrite the control estimate."""
from types import SimpleNamespace
import unittest
from sensor_msgs.msg import JointState
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode


class TestInputContract(unittest.TestCase):
    def state(self):
        return SimpleNamespace(_sim_door_feedback_joint_name='camera_door_rotation',
            _sim_lever_feedback_joint_name='lever_joint', _sim_door_position=.2,
            _sim_lever_position=0., _feedback_door_msg_stamp=(0,0),
            _feedback_lever_msg_stamp=(0,0), _feedback_door_stamp=0.,
            _feedback_lever_stamp=0., _feedback_lever_samples=[])

    def test_truth_angle_is_ignored_but_lever_is_retained(self):
        state=self.state()
        message=JointState()
        message.header.stamp.sec=1
        message.name=['door_hinge','lever_joint']
        message.position=[2.09,-.1]
        ManipulationNode._sim_door_state_callback(state,message)
        self.assertEqual(state._sim_door_position,.2)
        self.assertEqual(state._feedback_door_stamp,0.)
        self.assertEqual(state._sim_lever_position,-.1)

    def test_only_observed_rotation_updates_door_control(self):
        state=self.state()
        message=JointState()
        message.header.stamp.sec=1
        message.name=['camera_door_rotation']
        message.position=[.7]
        ManipulationNode._sim_door_state_callback(state,message)
        self.assertEqual(state._sim_door_position,.7)
        self.assertGreater(state._feedback_door_stamp,0.)

    def test_invalid_estimate_cannot_refresh_feedback(self):
        state=self.state()
        message=JointState()
        message.header.stamp.sec=1
        message.name=['camera_door_rotation']
        message.position=[float('nan')]
        ManipulationNode._sim_door_state_callback(state,message)
        self.assertEqual(state._feedback_door_stamp,0.)
        self.assertEqual(state._sim_door_position,.2)

    def test_continuing_truth_does_not_mask_camera_dropout(self):
        state=self.state()
        camera=JointState()
        camera.header.stamp.sec=1
        camera.name=['camera_door_rotation']
        camera.position=[.7]
        ManipulationNode._sim_door_state_callback(state,camera)
        last_update=state._feedback_door_stamp
        truth=JointState()
        truth.header.stamp.sec=2
        truth.name=['door_hinge','lever_joint']
        truth.position=[2.09,-.15]
        ManipulationNode._sim_door_state_callback(state,truth)
        self.assertEqual(state._sim_door_position,.7)
        self.assertEqual(state._feedback_door_stamp,last_update)
        self.assertEqual(state._feedback_door_msg_stamp,(1,0))


if __name__=='__main__':
    unittest.main()
