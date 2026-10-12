"""Read-only ROS adapter checks, independent of the active simulation."""
import math
from collections import deque
from types import SimpleNamespace as NS
import unittest
from builtin_interfaces.msg import Time as Stamp
from fire_robot_navigation.observed_footprint import ObservedFootprintGuard


class GuardSensorContractTests(unittest.TestCase):
    def make(self):
        guard = ObservedFootprintGuard.__new__(ObservedFootprintGuard)
        guard.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=100100000000)))
        guard.bounds = [-.245, .312, -.226, .226]
        guard.grid = None
        guard.free_history = deque(maxlen=20)
        guard.free_history_age_sec = 2.
        guard.static_history_ready = False
        guard.requests = []

        def lookup(target, source, at, **kwargs):
            guard.requests.append((target, source, at.nanoseconds))
            return NS(transform=NS(translation=NS(x=.25 if target=='base_link' else 0., y=0.),
                                   rotation=NS(x=0.,y=0.,z=0.,w=1.)))

        guard.tf = NS(lookup_transform=lookup)
        scan = NS(header=NS(stamp=Stamp(sec=100, nanosec=0), frame_id='lidar_link'),
                  angle_min=-math.pi, angle_increment=math.pi/360.,
                  range_min=.15, range_max=12., ranges=[3.]*721)
        return guard, scan

    def test_measurement_timestamp_used_for_all_transforms(self):
        guard, scan = self.make()
        self.assertIsNotNone(guard.snapshot(scan, 100.))
        self.assertEqual(guard.requests, [('base_link','lidar_link',100000000000),
                                          ('map','base_link',100000000000),
                                          ('odom','base_link',100000000000)])

    def test_repeated_snapshot_does_not_count_as_independent_free_space_evidence(self):
        guard,scan=self.make()
        guard.snapshot(scan,100.)
        scene=guard.snapshot(scan,100.)
        self.assertEqual(len(guard.free_history),1)
        self.assertEqual(len(scene.free_history),1)
        self.assertEqual(scene.measurement_stamp,100.)

    def test_extended_static_history_only_after_completed_manipulation(self):
        guard,scan=self.make()
        guard.free_history_age_sec=30.
        self.assertEqual(guard.snapshot(scan,100.).free_history_age_sec,2.)
        guard.on_phase(NS(data='COMPLETE:observed_door'))
        self.assertEqual(guard.snapshot(scan,100.).free_history_age_sec,30.)
        guard.on_phase(NS(data='PRESS_LEVER:next_door'))
        self.assertEqual(guard.snapshot(scan,100.).free_history_age_sec,2.)

    def test_old_received_sample_and_future_sample_are_rejected(self):
        for received, stamp in ((99.,100),(100.,99),(100.,101),(101.,100)):
            with self.subTest(received=received,stamp=stamp):
                guard, scan = self.make()
                scan.header.stamp.sec = stamp
                self.assertIsNone(guard.snapshot(scan, received))
                self.assertFalse(guard.requests)

    def test_stale_or_wrong_frame_grid_cannot_clear_blind_rays(self):
        for frame,stamp,receipt in [('odom',100,100.),('map',96,100.),('map',100,96.)]:
            with self.subTest(frame=frame,stamp=stamp,receipt=receipt):
                guard, scan = self.make()
                guard.grid = (NS(header=NS(frame_id=frame,stamp=NS(sec=stamp,nanosec=0))),receipt)
                self.assertIsNone(guard.snapshot(scan,100.).grid)

    def test_fresh_observed_map_is_accepted(self):
        guard, scan = self.make()
        grid = NS(header=NS(frame_id='map',stamp=NS(sec=100,nanosec=0)))
        guard.grid = (grid,100.)
        self.assertIs(guard.snapshot(scan,100.).grid,grid)


if __name__ == '__main__':
    unittest.main()
