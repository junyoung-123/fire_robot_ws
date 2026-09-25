"""Fail if the angle adapter changes the preserved successful motion policy."""
import hashlib
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from sensor_msgs.msg import JointState
from fire_robot_manipulation import preserved_contact_policy
from fire_robot_manipulation.minimal_observed_angle_node import MinimalObservedAngle


class TestMinimalPolicy(unittest.TestCase):
    def test_preserved_file_is_byte_identical_to_successful_source(self):
        value=hashlib.sha256(Path(preserved_contact_policy.__file__).read_bytes()).hexdigest()
        self.assertEqual(value,'ea9a63ec1d7c26fff740d62208abf9e0d9ec01cb0c83b2ee530a9c185b85ac0a')

    def test_only_input_and_diagnostic_methods_overridden(self):
        methods={name for name,value in MinimalObservedAngle.__dict__.items() if inspect.isfunction(value)}
        self.assertEqual(methods,{'__init__','_feedback_fresh','_feedback_event'})

    def test_motion_functions_are_original_objects(self):
        for name in ('_execute_door_open_sequence','_feedback_press','_feedback_handle_held_open',
                     '_feedback_probe_contact_motion','_feedback_retract_released_handle',
                     '_move_sim_arm_to_stow','_move_to_home','_post_open_backoff'):
            self.assertIs(getattr(MinimalObservedAngle,name),getattr(preserved_contact_policy.ManipulationNode,name))

    def test_raw_simulator_angle_cannot_update_control_angle(self):
        state=SimpleNamespace(_sim_door_feedback_joint_name='camera_door_rotation',
            _sim_lever_feedback_joint_name='lever_joint',_sim_door_position=.2,
            _sim_lever_position=0.,_feedback_door_msg_stamp=(0,0),
            _feedback_lever_msg_stamp=(0,0),_feedback_door_stamp=0.,
            _feedback_lever_stamp=0.,_feedback_lever_samples=[])
        message=JointState()
        message.name=['door_hinge','lever_joint']
        message.position=[2.1,-.1]
        message.header.stamp.sec=1
        MinimalObservedAngle._sim_door_state_callback(state,message)
        self.assertEqual(state._sim_door_position,.2)
        self.assertEqual(state._feedback_door_stamp,0.)
        self.assertEqual(state._sim_lever_position,-.1)


if __name__=='__main__': unittest.main()
