from collections import deque
import json
from types import SimpleNamespace
import unittest
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class AtomicPanelObservationTests(unittest.TestCase):
    def stub(self):
        received = []
        stub = SimpleNamespace(_contact_reference_frame='contact_scan',
            _plane_confirmations=deque(), _sim_door_state_callback=received.append)
        return stub, received

    def test_confidence_is_present_before_angle_processing(self):
        stub, received = self.stub()
        def consume(measurement):
            self.assertEqual(len(stub._plane_confirmations), 1)
            self.assertEqual(stub._plane_confirmations[-1]['stamp'], 12.5)
            received.append(measurement)
        stub._sim_door_state_callback=consume
        payload = dict(valid=True, sensor='rgbd', reference_frame='contact_scan',
                       depth_stamp=12.5, angle_rad=.3, orientation_uncertainty_rad=.01)
        RobotObservationManipulation._panel_status_callback(stub, SimpleNamespace(data=json.dumps(payload)))
        self.assertEqual(len(stub._plane_confirmations), 1)
        self.assertEqual(stub._plane_confirmations[-1]['stamp'], 12.5)
        self.assertEqual(received[0].header.stamp.sec, 12)
        self.assertEqual(received[0].header.stamp.nanosec, 500000000)
        self.assertEqual(received[0].name, ['camera_door_rotation'])
        self.assertEqual(list(received[0].position), [.3])

    def test_wrong_frame_invalid_or_environmental_input_cannot_feed_control(self):
        for payload in (dict(valid=False),
                        dict(valid=True, sensor='gazebo', reference_frame='contact_scan'),
                        dict(valid=True, sensor='rgbd', reference_frame='evaluation')):
            stub, received = self.stub()
            RobotObservationManipulation._panel_status_callback(stub, SimpleNamespace(data=json.dumps(payload)))
            self.assertFalse(received)
            self.assertFalse(stub._plane_confirmations)


if __name__ == '__main__':
    unittest.main()
