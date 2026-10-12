import math
from collections import deque
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import numpy as np
from fire_robot_navigation.observed_footprint import (
    historical_ray_support, ObservedSweep, ObservedFootprintGuard)


class StaticReturnMemoryTests(unittest.TestCase):
    def scan(self, ranges=None):
        return NS(ranges=[3.]*721 if ranges is None else ranges,
                  angle_min=-math.pi, angle_increment=math.pi/360,
                  range_min=.1, range_max=12.)

    def history(self, pose=(0.,0.,0.)):
        return [(10.,pose,(.25,0.,0.),self.scan()),
                (12.,pose,(.25,0.,0.),self.scan())]

    def support(self, *, now=20., pose=(0.,0.,0.), age=30., history=None):
        return historical_ray_support(np.array([[.5,0.]]), np.array(pose), now,
            self.history() if history is None else history, max_age_sec=age)

    def test_old_free_space_requires_explicit_static_option(self):
        self.assertFalse(self.support(age=2.).all())
        self.assertTrue(self.support().all())

    def test_stale_single_and_repeated_views_do_not_clear_space(self):
        self.assertFalse(self.support(now=43.).any())
        self.assertFalse(self.support(history=self.history()[:1]).any())
        repeated=[self.history()[0]]*10
        self.assertFalse(self.support(history=repeated).any())

    def test_excess_odometry_separation_rejects_old_proof(self):
        for pose in ((.51,0.,0.),(0.,0.,math.pi/2+.01)):
            self.assertFalse(self.support(pose=pose).any())

    def test_invalid_memory_horizon_does_not_clear_space(self):
        for age in (math.nan, math.inf, 31., 0.):
            self.assertFalse(self.support(age=age).any())

    def test_current_obstacle_shadow_overrides_old_free_observation(self):
        ranges=[3.]*721
        ranges[359:362]=[.5,math.nan,.5]
        scene=ObservedSweep(self.scan(ranges),(.25,0.,0.),[-.245,.312,-.226,.226],
            (0.,0.,0.),None,free_history=self.history(),odom_pose=np.zeros(3),
            measurement_stamp=20.,free_history_age_sec=30.)
        self.assertFalse(scene.observed_free(np.array([[1.,0.]])).any())
        ranges[359:362]=[math.nan]*3
        scene=ObservedSweep(self.scan(ranges),(.25,0.,0.),scene.bounds,scene.pose,None,
            free_history=self.history(),odom_pose=np.zeros(3),measurement_stamp=20.,
            free_history_age_sec=30.)
        self.assertTrue(scene.observed_free(np.array([[1.,0.]])).all())

    def test_moving_door_transition_clears_cached_free_space(self):
        guard=ObservedFootprintGuard.__new__(ObservedFootprintGuard)
        guard.free_history=deque(self.history())
        guard.static_history_ready=False
        guard.on_phase(NS(data='COMPLETE:observed_id'))
        self.assertTrue(guard.static_history_ready)
        self.assertEqual(len(guard.free_history),0)
        guard.free_history.extend(self.history())
        guard.on_phase(NS(data='COMPLETE:observed_id'))
        self.assertEqual(len(guard.free_history),2)
        guard.on_phase(NS(data='PRE_GRASP:next_observed_id'))
        self.assertFalse(guard.static_history_ready)
        self.assertEqual(len(guard.free_history),0)

    def test_route_includes_fine_primitives_and_stops_at_first_useful_depth(self):
        scene=ObservedSweep(self.scan(),(.25,0.,0.),[-.245,.312,-.226,.226],(0.,0.,0.),None)
        scene.safe=Mock(side_effect=lambda v,w,**kw: 0.<v<=.03 and w==0.)
        result=scene._lookahead_recovery(0.,.06,.5,depth_limit=12,beam_width=10)
        self.assertEqual(result,(.03,0.))
        self.assertEqual(len(scene.recovery_path),2)
        self.assertGreater(scene.recovery_path[-1][2][0],.06)
        self.assertTrue(all(v<=.03 and w==0. for v,w,_ in scene.recovery_path))


if __name__=='__main__':
    unittest.main()
