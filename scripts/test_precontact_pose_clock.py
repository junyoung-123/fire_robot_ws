import inspect
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from fire_robot_manipulation import mission_contact_adapter as module


def transform(seconds, nanos=0):
    return NS(header=NS(stamp=NS(sec=seconds, nanosec=nanos)))


class PrecontactPoseClockTests(unittest.TestCase):
    def run_wait(self, samples, failures=None):
        node = NS(now=Mock(return_value=10.), command=Mock(),
                  tf=NS(lookup_transform=Mock(side_effect=samples)))
        deadline = Mock()
        deadline.failure.side_effect = failures or [None]*len(samples)
        with patch.object(module, 'SimMotionDeadline', return_value=deadline), \
                patch.object(module.rclpy, 'ok', return_value=True), \
                patch.object(module.time, 'sleep'):
            result = module.MissionContactAdapter.wait_precontact_pose(node)
        return result, node

    def test_future_tf_is_retried_not_restamped(self):
        future, valid = transform(10, 18000000), transform(9, 990000000)
        result, node = self.run_wait([future, valid])
        self.assertIs(result, valid)
        self.assertEqual(future.header.stamp.nanosec, 18000000)
        self.assertEqual(node.command.call_count, 2)
        for call in node.command.call_args_list:
            self.assertEqual(call.args, ())

    def test_stale_tf_and_missing_tf_wait_for_new_measurement(self):
        valid = transform(9, 950000000)
        result, node = self.run_wait([
            module.TransformException('not yet available'), transform(8), valid])
        self.assertIs(result, valid)
        self.assertEqual(node.tf.lookup_transform.call_count, 3)

    def test_timeout_never_returns_old_pose(self):
        with self.assertRaisesRegex(ValueError, 'Fresh observed precontact'):
            self.run_wait([transform(8)], [None, 'simulation deadline exceeded'])

    def test_original_age_boundary_is_unchanged(self):
        valid = transform(9, 250000000)
        result, _ = self.run_wait([transform(9, 249999999), valid])
        self.assertIs(result, valid)

    def test_latest_handle_is_rechecked_after_pose_wait(self):
        source = inspect.getsource(module.MissionContactAdapter.open)
        wait = source.index('parked = self.wait_precontact_pose()')
        handle = source.index('observed, exposure, anchored=self.latest', wait)
        recheck = source.index('self.current_handle_xyz(self.latest)', handle)
        service = source.index('self.client.call_async(physical)', recheck)
        self.assertLess(wait, handle)
        self.assertLess(recheck, service)


if __name__ == '__main__':
    unittest.main()
