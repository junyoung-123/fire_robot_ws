import math
import unittest
from collections import deque
from unittest.mock import Mock
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from fire_robot_navigation.motion_scan_filter import MotionGate, MotionScanFilter


class MotionGateTest(unittest.TestCase):
    def test_initial_scan_then_stationary_noise_not_accumulated(self):
        gate = MotionGate()
        self.assertTrue(gate.accept((0., 0., 0.)))
        for _ in range(1000):
            self.assertFalse(gate.accept((.0001, -.0001, .0001)))

    def test_in_place_turn_is_not_dropped(self):
        gate = MotionGate()
        gate.accept((0., 0., 0.))
        self.assertTrue(gate.accept((0., 0., .04)))

    def test_small_increments_accumulate_from_last_emitted_pose(self):
        gate = MotionGate()
        gate.accept((0., 0., 0.))
        self.assertFalse(gate.accept((.02, 0., 0.)))
        self.assertTrue(gate.accept((.04, 0., 0.)))

    def test_wraparound_and_invalid_pose(self):
        gate = MotionGate()
        gate.accept((0., 0., math.pi-.001))
        self.assertFalse(gate.accept((0., 0., -math.pi+.001)))
        self.assertFalse(gate.accept((float('nan'), 0., 0.)))

    def test_startup_retries_until_map_and_transform_are_received(self):
        node = object.__new__(MotionScanFilter)
        node.gate = MotionGate()
        node.odometry = deque()
        node.last_scan_time = None
        node.map_ready = False
        node.tf_ready = False
        node.buffer = Mock()
        node.buffer.lookup_transform.side_effect = TransformException('not ready')
        node.publisher = Mock()
        node.publisher.get_subscription_count.return_value = 1
        for t in range(1, 6):
            scan = LaserScan()
            scan.header.stamp.sec = t
            node.odometry.append((t, (0., 0., 0.)))
            node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 5)
        grid = OccupancyGrid()
        grid.info.width = grid.info.height = 1
        node.map_callback(grid)
        scan.header.stamp.sec = 6
        node.odometry.append((6, (0., 0., 0.)))
        node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 6)
        node.buffer.lookup_transform.side_effect = None
        transform = TransformStamped()
        transform.header.stamp.sec = 7
        node.buffer.lookup_transform.return_value = transform
        scan.header.stamp.sec = 7
        node.odometry.append((7, (0., 0., 0.)))
        node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 6)
        scan.header.stamp.sec = 8
        transform.header.stamp.sec = 8
        node.odometry.append((8, (0., 0., .05)))
        node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 7)
        # The robot stops after a later localization outage. Old TF must not
        # permanently latch readiness and suppress the recovery scans.
        scan.header.stamp.sec = 9
        node.odometry.append((9, (0., 0., .05)))
        node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 8)
        self.assertIs(node.publisher.publish.call_args.args[0], scan)
        scan.header.stamp.nanosec = 100000000
        node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 8)
        scan.header.stamp.sec = 10
        scan.header.stamp.nanosec = 0
        transform.header.stamp.sec = 10
        node.odometry.append((10, (0., 0., .05)))
        node.scan_callback(scan)
        self.assertEqual(node.publisher.publish.call_count, 8)


if __name__ == '__main__':
    unittest.main()
