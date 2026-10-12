import unittest
from collections import deque
from unittest.mock import Mock,patch
import numpy as np
from fire_robot_interfaces.msg import DoorInfo
from geometry_msgs.msg import PointStamped
from fire_robot_manipulation.mission_contact_adapter import MissionContactAdapter
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics
from fire_robot_perception.handle_image_tracker import HandleImageTracker


class PartialHandleTrackingTests(unittest.TestCase):
    def tracker(self):
        tracker=HandleImageTracker()
        tracker.gray=np.zeros((100,100),np.uint8)
        tracker.box=np.array([[3.,30.],[43.,60.]],np.float32)
        tracker.points=np.array([[[10.,35.]],[[15.,35.]],[[25.,55.]],[[35.,55.]]],np.float32)
        tracker.seed_time=1.
        tracker.confidence=.8
        return tracker

    def test_partial_roi_keeps_measured_original_centre_inside_view(self):
        tracker=self.tracker(); old=tracker.points.copy(); moved=old+[-5.,0.]
        status=np.ones((4,1),np.uint8)
        matrix=np.array([[1.,0.,-5.],[0.,1.,0.]])
        with patch('cv2.calcOpticalFlowPyrLK',side_effect=[(moved,status,None),(old,status,None)]), \
                patch('cv2.estimateAffinePartial2D',return_value=(matrix,status)):
            box,quality=tracker.advance(np.zeros((100,100,3),np.uint8),2.)
        self.assertEqual(box,(-2,30,38,60))
        self.assertEqual(quality,.8)
        before=tracker.box.copy()
        self.assertTrue(tracker.confirm_partial_primary((0,30,38,60),.7,2.))
        np.testing.assert_array_equal(tracker.box,before)
        self.assertEqual(tracker.seed_time,2.)
        self.assertEqual(tracker.confidence,.7)

    def test_out_of_view_grasp_centre_is_still_rejected(self):
        tracker=self.tracker(); old=tracker.points.copy(); moved=old+[-25.,0.]
        status=np.ones((4,1),np.uint8)
        matrix=np.array([[1.,0.,-25.],[0.,1.,0.]])
        with patch('cv2.calcOpticalFlowPyrLK',side_effect=[(moved,status,None),(old,status,None)]), \
                patch('cv2.estimateAffinePartial2D',return_value=(matrix,status)):
            self.assertIsNone(tracker.advance(np.zeros((100,100,3),np.uint8),2.))

    def test_wrong_object_low_confidence_and_expired_track_cannot_refresh(self):
        for box,confidence,now in (((70,30,95,60),.8,2.),((3,30,43,60),.44,2.),
                                   ((3,30,43,60),.8,17.)):
            tracker=self.tracker()
            self.assertFalse(tracker.confirm_partial_primary(box,confidence,now))
            self.assertEqual(tracker.seed_time,1.)

    def test_current_partial_yolo_cannot_reseed_to_its_shifted_centre(self):
        from fire_robot_perception.door_detection_node import DoorDetectionNode,HandleObservation
        node=DoorDetectionNode.__new__(DoorDetectionNode)
        node._detect_handle_enabled=True
        node._handle_registered_depth=True
        node._handle_yolo_fallback_hsv=False
        tracker=self.tracker()
        tracker.box=np.array([[-2.,30.],[38.,60.]],np.float32)
        tracker.advance=Mock(return_value=((-2,30,38,60),.8))
        tracker.seed=Mock()
        node._feedback_handle_trackers={'front':tracker}
        node._detect_yolo_handle_observation=Mock(return_value=HandleObservation(
            x=19,y=45,confidence=.7,method='yolo:primary:item',bbox=(0,30,38,60)))
        image=np.zeros((100,100,3),np.uint8)
        with patch('time.monotonic',return_value=2.):
            observed=node._detect_handle_observation(image,image,0,0,100,100,'blue',image.shape)
        self.assertEqual(observed.x,18)
        self.assertTrue(observed.method.startswith('track:primary_yolo'))
        tracker.seed.assert_not_called()


class CameraReacquisitionTests(unittest.TestCase):
    def node(self):
        node = MissionContactAdapter.__new__(MissionContactAdapter)
        node.reference_frame = 'odom'
        node.active = True
        node.request_target = PointStamped()
        node.request_target.point.x = 6.
        node.request_target.point.y = -1.8
        node.selected = Mock()
        node.latest = None
        node.handle_positions=deque(maxlen=12)
        node.now = Mock(return_value=5.)
        node.command = Mock()
        node.safe_motion = Mock(return_value=True)
        node.kinematics = PiperActualKinematics()
        return node

    def point(self, x, y=0., z=.75):
        point = PointStamped()
        point.header.stamp.sec = 5
        point.point.x, point.point.y, point.point.z = x, y, z
        return point

    def test_confidence_gate_unchanged(self):
        node = self.node()
        msg = DoorInfo()
        msg.handle_detected = True
        msg.handle_detection_method = 'yolo:primary:item:registered_depth'
        msg.handle_confidence = .358
        node.transform = Mock(side_effect=[self.point(.93, .44), self.point(6., -1.8, .82)])
        node.observation(msg, handle_only=True)
        self.assertIsNotNone(node.view_hint)
        self.assertIsNone(node.latest)
        self.assertIsNone(node.handle_track.anchor)
        node.selected.publish.assert_not_called()

    def test_association_uses_contact_reference_not_shifting_map(self):
        node = self.node()
        msg = DoorInfo()
        msg.handle_detected = True
        msg.handle_detection_method = 'yolo:primary:item:registered_depth'
        msg.handle_confidence = .77
        node.transform = Mock(side_effect=[self.point(.62, .2), self.point(6., -1.8, .82)])
        node.observation(msg, handle_only=True)
        self.assertEqual([call.args[1] for call in node.transform.call_args_list], ['base_link', 'odom'])
        self.assertIsNotNone(node.latest)
        node.selected.publish.assert_called_once()

    def test_retreat_needs_rear_scan_and_remains_within_real_arm_reach(self):
        node = self.node()
        latest = (None, 1., np.array([6., -1.8, .82]))
        node.transform = Mock(return_value=self.point(.58, .18))
        self.assertTrue(node.reacquire_camera(latest))
        node.command.assert_called_with(-.02, 0.)
        node.safe_motion.return_value = False
        self.assertFalse(node.reacquire_camera(latest))
        node.command.assert_called_with()
        node.safe_motion.return_value = True
        node.transform.return_value = self.point(2., .18)
        self.assertFalse(node.reacquire_camera(latest))
        node.command.assert_called_with()

    def test_no_observed_anchor_no_blind_recovery(self):
        node = self.node()
        self.assertFalse(node.reacquire_camera(None))
        node.command.assert_called_with()


if __name__ == '__main__':
    unittest.main()
