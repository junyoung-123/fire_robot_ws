"""Blind-ray support requires measured free space, not a default-free map."""
import math
from types import SimpleNamespace as NS
import unittest
import numpy as np
from fire_robot_navigation.observed_footprint import ObservedSweep


class LocalObservationMemoryTests(unittest.TestCase):
    def grid(self, cost=0):
        return NS(data=[cost]*1600,info=NS(width=40,height=40,resolution=.1,
            origin=NS(position=NS(x=-2.,y=-2.),orientation=NS(x=0.,y=0.,z=0.,w=1.))))

    def scene(self, local=None, global_grid=None, odom=(0.,0.,0.)):
        scan=NS(ranges=[3.]*721,angle_min=-math.pi,angle_increment=math.pi/360,
                range_min=.1,range_max=12.)
        scan.ranges[:90]=[math.nan]*90
        scan.ranges[-90:]=[math.nan]*90
        return ObservedSweep(scan,(.25,0.,0.),[-.245,.312,-.226,.226],
            (20.,20.,0.),global_grid,odom_pose=odom,local_grid=local)

    def test_own_odom_memory_not_slam_pose_fills_blind_ray(self):
        scene=self.scene(self.grid())
        self.assertTrue(scene.observed_free(np.array([[-.5,0.]])).all())
        self.assertTrue(scene.safe(-.03,0.))

    def test_absent_unknown_inscribed_lethal_never_fill_blind_ray(self):
        for grid in (None,self.grid(-1),self.grid(99),self.grid(100)):
            self.assertFalse(self.scene(grid).safe(-.03,0.))

    def test_missing_or_wrong_odom_cannot_use_local_memory(self):
        for odom in (None,(20.,20.,0.),(math.nan,0.,0.)):
            self.assertFalse(self.scene(self.grid(),odom=odom).safe(-.03,0.))

    def test_observed_inflation_below_inscribed_cost_is_allowed(self):
        self.assertTrue(self.scene(self.grid(98)).safe(-.03,0.))

    def test_current_obstacle_and_neighbor_shadow_override_memory(self):
        scene=self.scene(self.grid())
        scene.ray_limits[:2]=.3
        scene.ray_limits[-2:]=.3
        self.assertFalse(scene.observed_free(np.array([[-.5,0.]])).all())

    def test_local_obstacle_overrides_old_global_free_cell(self):
        scene=self.scene(self.grid(100),self.grid())
        scene.pose=np.zeros(3)
        self.assertFalse(scene.observed_free(np.array([[-.5,0.]])).all())

    def test_new_local_observation_can_resolve_old_global_occupancy(self):
        scene=self.scene(self.grid(),self.grid(100))
        scene.pose=np.zeros(3)
        self.assertTrue(scene.observed_free(np.array([[-.5,0.]])).all())

    def test_bad_grid_geometry_never_fills_a_blind_ray(self):
        grid=self.grid();grid.info.origin.orientation.w=0.
        self.assertFalse(self.scene(grid).safe(-.03,0.))

    def test_global_inflated_unknown_is_not_observed_free(self):
        scene=self.scene(None,self.grid(99));scene.pose=np.zeros(3)
        self.assertFalse(scene.safe(-.03,0.))

    def test_local_obstacle_overrides_historical_blind_ray_support(self):
        scene=self.scene(self.grid(100))
        old=NS(ranges=[3.]*721,angle_min=-math.pi,angle_increment=math.pi/360,
               range_min=.1,range_max=12.)
        scene.free_history=[(9.5,(0.,0.,0.),(.25,0.,0.),old),
                            (9.7,(0.,0.,0.),(.25,0.,0.),old)]
        scene.measurement_stamp=10.
        self.assertFalse(scene.observed_free(np.array([[-.5,0.]])).all())


if __name__=='__main__':unittest.main()
