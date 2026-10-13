"""Recorded sensor-only regression; this is not a complete mission test."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from fire_robot_navigation.observed_footprint import ObservedSweep


class RecordedReturnCornerTests(unittest.TestCase):
    def test_checked_detour_is_found_without_revisiting_shorter_search_states(self):
        fixture=Path(__file__).parent/'fixtures/observed_return_corner.npz'
        with np.load(fixture) as data:
            spec=dict(zip(('angle_min','angle_increment','range_min','range_max'),data['scan_spec']))
            def grid(name):
                cells=data[name+'_data'];origin=data[name+'_origin']
                return NS(data=cells.ravel(),info=NS(width=cells.shape[1],height=cells.shape[0],
                    resolution=float(data[name+'_resolution']),origin=NS(
                    position=NS(x=float(origin[0]),y=float(origin[1])),
                    orientation=NS(x=0.,y=0.,z=0.,w=1.))))
            history=[(float(t),p,s,NS(ranges=r,**spec)) for t,p,s,r in zip(
                data['history_stamps'],data['history_poses'],data['history_sensors'],data['history_ranges'])]
            scene=ObservedSweep(NS(ranges=data['ranges'],**spec),data['sensor'],
                [-.245,.312,-.226,.226],data['map_pose'],grid('global'),
                free_history=history,odom_pose=data['odom_pose'],
                measurement_stamp=float(data['measurement_stamp']),free_history_age_sec=30.,
                local_grid=grid('local'))
            self.assertFalse(scene.safe(.06,0.))
            path=scene.extended_recovery_path(0.)
            self.assertIsNotNone(path)
            self.assertGreater(path[-1][2][0],.12)
            start=np.zeros(3)
            for v,w,end in path:
                self.assertTrue(scene.safe(v,w,start=start),scene.last_failure)
                start=end


if __name__=='__main__':
    unittest.main()
