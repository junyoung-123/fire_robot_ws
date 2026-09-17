import unittest
from threading import Lock
from types import SimpleNamespace as NS
from unittest.mock import Mock

from fire_robot_perception.door_detection_node import DoorDetectionNode


class PerceptionCallbackSchedulingTests(unittest.TestCase):
    def make_node(self):
        node = object.__new__(DoorDetectionNode)
        node._image_processing_lock = Lock()
        node._image_interval_for_source = lambda _: 0.
        node._last_image_process_time = {}
        node.bridge = NS(imgmsg_to_cv2=Mock(return_value='decoded'))
        node._process = Mock()
        return node

    def test_busy_detector_drops_frame_without_waiting_or_decoding(self):
        node = self.make_node()
        node._image_processing_lock.acquire()
        try:
            node.image_callback(NS(header='stamp'), NS(name='front'))
            node.bridge.imgmsg_to_cv2.assert_not_called()
            node._process.assert_not_called()
        finally:
            node._image_processing_lock.release()

    def test_frame_keeps_exposure_header(self):
        node = self.make_node()
        source = NS(name='front')
        node.image_callback(NS(header='original stamp'), source)
        node._process.assert_called_once_with('decoded', 'original stamp', source)
        self.assertFalse(node._image_processing_lock.locked())

    def test_failed_inference_does_not_lock_future_frames(self):
        node = self.make_node()
        node._process.side_effect = RuntimeError('inference failed')
        with self.assertRaises(RuntimeError):
            node.image_callback(NS(header='stamp'), NS(name='front'))
        self.assertFalse(node._image_processing_lock.locked())

    def test_throttled_frame_releases_lock(self):
        node = self.make_node()
        node._image_interval_for_source = lambda _: 10.
        import time
        node._last_image_process_time['front'] = time.monotonic()
        node.image_callback(NS(header='stamp'), NS(name='front'))
        node._process.assert_not_called()
        self.assertFalse(node._image_processing_lock.locked())


if __name__ == '__main__':
    unittest.main()
