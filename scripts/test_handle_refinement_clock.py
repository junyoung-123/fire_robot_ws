from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode


class HandleRefinementClockTests(unittest.TestCase):
    def node(self, stamps, received):
        node = object.__new__(ManipulationNode)
        node._feedback_contact = True
        node._sim_time_sec = lambda: 20.
        node._detected_handle_samples = [
            (index, NS(header=NS(stamp=NS(sec=sec, nanosec=nsec))), .8, 'yolo:primary:registered_depth')
            for index, (sec, nsec) in enumerate(stamps, 1)]
        node._feedback_handle_times = dict(enumerate(received, 1))
        return node

    def samples(self, node):
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.time.monotonic', return_value=10.):
            return node._fresh_refinement_samples(0)

    def test_slow_simulation_preserves_three_fresh_exposures(self):
        node = self.node([(18, 800000000), (19, 400000000), (20, 0)], [6.6, 8.3, 10.])
        self.assertEqual(len(self.samples(node)), 3)

    def test_recent_delivery_does_not_refresh_old_images(self):
        node = self.node([(16, 0), (16, 100000000), (16, 200000000)], [9.8, 9.9, 10.])
        self.assertEqual(self.samples(node), [])

    def test_duplicates_are_not_three_observations(self):
        node = self.node([(20, 0)] * 3, [9.8, 9.9, 10.])
        self.assertEqual(len(self.samples(node)), 1)

    def test_dead_stream_fails_even_if_simulation_clock_is_paused(self):
        node = self.node([(19, 0), (19, 500000000), (20, 0)], [4., 5., 6.])
        self.assertEqual(self.samples(node), [])

    def test_zero_and_future_timestamps_are_rejected(self):
        node = self.node([(0, 0), (21, 0), (20, 0)], [9.8, 9.9, 10.])
        self.assertEqual(len(self.samples(node)), 1)


if __name__ == '__main__':
    unittest.main()
