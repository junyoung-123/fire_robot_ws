import unittest
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode


class ContactSensorPauseBoundaryTests(unittest.TestCase):
    def test_recovered_sensor_wait_aborts_step_before_old_arm_base_command_resumes(self):
        node=ManipulationNode.__new__(ManipulationNode)
        node._feedback_config={'follow_step_m':.008}
        node._feedback_measurement_pause_generation=0
        node._feedback_retain_lever_press=Mock(return_value=True)
        node._feedback_follow_contact=Mock(return_value=True)
        node._feedback_settle_held_target=Mock(return_value=True)
        node._feedback_tool=Mock(return_value=np.array([.6,0.,.7]))
        node._sim_time_sec=Mock(return_value=0.)
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        def recovered():
            node._feedback_measurement_pause_generation+=1
            return True
        node._feedback_fresh=recovered
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok',return_value=True):
            self.assertEqual(node._feedback_probe_contact_motion(speed=.02),'sensor_wait')
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
        self.assertTrue(node._feedback_event.call_args.kwargs['interrupted_for_sensor_wait'])


if __name__=='__main__':
    unittest.main()
