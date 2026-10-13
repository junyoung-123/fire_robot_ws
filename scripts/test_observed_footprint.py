import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from concurrent.futures import Future
import numpy as np
from fire_robot_navigation.observed_footprint import (
    ObservedSweep, ObservedFootprintGuard, arc_pose,historical_ray_support)


class ObservedFootprintTests(unittest.TestCase):
    def scene(self, ranges=None, grid=None):
        scan = NS(ranges=[3.]*721 if ranges is None else ranges,
                  angle_min=-math.pi, angle_increment=math.pi/360,
                  range_min=.1, range_max=12.)
        return ObservedSweep(scan, (.25, 0., 0.), [-.245, .312, -.226, .226], (0., 0., 0.), grid)

    def grid(self, cost):
        info = NS(resolution=.025, width=160, height=160,
                  origin=NS(position=NS(x=-2., y=-2.), orientation=NS(x=0., y=0., z=0., w=1.)))
        return NS(info=info, data=[cost]*25600)

    def test_clear_scan_allows_forward_reverse_and_turn(self):
        scene = self.scene()
        for v, w in ((.06, 0.), (-.06, 0.), (0., .24), (0., -.24)):
            self.assertTrue(scene.safe(v, w))

    def test_body_collision_cannot_be_overridden_by_free_grid(self):
        ranges = [3.]*721
        ranges[360] = .1
        scene = self.scene(ranges, self.grid(0))
        self.assertFalse(scene.safe(.06, 0.))

    def test_near_front_does_not_block_safe_reverse(self):
        ranges = [3.]*721
        ranges[360] = .17
        scene = self.scene(ranges)
        self.assertFalse(scene.safe(.06, 0.))
        self.assertTrue(scene.safe(-.06, 0.))

    def test_tight_contact_stance_can_only_increase_clearance(self):
        ranges = [3.]*721
        ranges[360] = .105
        scene = self.scene(ranges)
        self.assertLess(scene.clearance([0.,0.,0.]), .05)
        self.assertTrue(scene.safe(-.06, 0.))
        self.assertFalse(scene.safe(.06, 0.))
        ranges[360] = .085
        scan = scene.scan
        scan.range_min = .05
        scan.ranges = ranges
        scene = ObservedSweep(scan, (.25,0.,0.), [-.245,.312,-.226,.226], (0.,0.,0.), self.grid(0))
        self.assertFalse(scene.safe(-.06, 0.))

    def test_free_map_cannot_clear_space_behind_a_current_obstacle(self):
        ranges = [3.]*721
        ranges[359:362] = [.5]*3
        scene = self.scene(ranges, self.grid(0))
        self.assertFalse(scene.observed_free(np.array([[1., 0.]])).all())

    def close_front_scene(self, gap=.015, grid=None):
        ranges = [3.]*721
        ranges[360] = .312-.25+gap
        scan = NS(ranges=ranges, angle_min=-math.pi,
                  angle_increment=math.pi/360, range_min=.01, range_max=12.)
        return ObservedSweep(scan, (.25,0.,0.), [-.245,.312,-.226,.226], (0.,0.,0.), grid)

    def test_close_but_nonintersecting_front_allows_bounded_withdrawal(self):
        scene = self.close_front_scene()
        self.assertTrue(scene.safe(-.03, 0.))
        self.assertEqual(scene.choose(1.)[:2], (-.03, 0.))
        self.assertGreater(scene.clearance(arc_pose(-.03,0.,1.5)), .03)
        for v,w in ((-.06,0.), (.03,0.), (-.03,.01), (0.,.12)):
            self.assertFalse(scene.safe(v,w))
        self.assertFalse(scene.safe(-.03,0.,horizon=2.))
        self.assertFalse(scene.safe(-.03,0.,start=[-.01,0.,0.]))

    def test_bounded_withdrawal_never_overrides_body_intersection(self):
        scene = self.close_front_scene(gap=-.01, grid=self.grid(0))
        self.assertFalse(scene.safe(-.03,0.))
        self.assertEqual(scene.choose(0.)[:2], (0.,0.))

    def test_tight_stance_tries_shorter_clearance_increasing_sweep(self):
        scene=self.close_front_scene()
        scene.safe=Mock(side_effect=lambda v,w: v==-.015 and w==0.)
        self.assertEqual(scene.choose(0.)[:2],(-.015,0.))
        scene.safe.assert_any_call(-.03,0.)
        scene.safe.assert_any_call(-.015,0.)

    def test_smallest_withdrawal_never_skips_the_same_safety_predicate(self):
        scene=self.close_front_scene()
        scene.safe=Mock(side_effect=lambda v,w: v==-.0075 and w==0.)
        self.assertEqual(scene.choose(0.)[:2],(-.0075,0.))
        self.assertEqual(scene.safe.call_count,3)

    def test_small_withdrawal_still_requires_clearance_gain(self):
        scene=self.scene()
        scene.obstacles=np.array([[0.,-.241],[3.,0.],[-3.,0.]])
        for reverse in (-.03,-.015,-.0075):
            self.assertFalse(scene.safe(reverse,0.))
        self.assertEqual(scene.choose(0.)[:2],(0.,0.))

    def test_bounded_withdrawal_still_requires_observed_rear(self):
        for cost in (None,-1,100):
            scene = self.close_front_scene(grid=None if cost is None else self.grid(cost))
            scene.scan.ranges[:90] = [math.nan]*90
            scene.scan.ranges[-90:] = [math.nan]*90
            scene = ObservedSweep(scene.scan, scene.sensor, scene.bounds, scene.pose, scene.grid)
            self.assertFalse(scene.safe(-.03,0.))

    def test_close_side_only_does_not_allow_unchanged_clearance_withdrawal(self):
        scene = self.scene()
        scene.obstacles = np.array([[0.,-.241],[3.,0.],[-3.,0.]])
        self.assertFalse(scene.safe(-.03,0.))
        self.assertEqual(scene.last_failure, 'WITHDRAWAL_DOES_NOT_GAIN_CLEARANCE')

    def test_close_front_with_rear_obstacle_never_allows_withdrawal(self):
        scene = self.close_front_scene()
        scene.obstacles = np.vstack((scene.obstacles,[-.27,0.]))
        self.assertFalse(scene.safe(-.03,0.))

    def test_missing_neighbor_cannot_hide_a_measured_obstacle(self):
        for rays in ([.5, math.nan, 3.], [math.nan, .5, 3.], [3., math.nan, .5]):
            with self.subTest(rays=rays):
                ranges = [3.]*721
                ranges[359:362] = rays
                scene = self.scene(ranges, self.grid(0))
                self.assertFalse(scene.observed_free(np.array([[1., 0.]])).all())
                self.assertTrue(scene.observed_free(np.array([[.4, 0.]])).all())

    def test_nan_blind_region_requires_known_observed_space(self):
        ranges = [3.]*721
        ranges[:90] = [math.nan]*90
        ranges[-90:] = [math.nan]*90
        self.assertFalse(self.scene(ranges).safe(-.06, 0.))
        self.assertTrue(self.scene(ranges, self.grid(0)).safe(-.06, 0.))
        for cost in (-1, 100):
            self.assertFalse(self.scene(ranges, self.grid(cost)).safe(-.06, 0.))

    def test_unknown_grid_never_clears_nan(self):
        scene = self.scene([math.nan]*90+[3.]*631, self.grid(-1))
        self.assertFalse(scene.observed_free(np.array([[-.5, -.1]])).all())

    def test_angular_sector_does_not_replace_lateral_body_extent(self):
        ranges = [3.]*721
        ranges[540] = .29
        scene = self.scene(ranges)
        self.assertFalse(scene.safe(0., .5))

    def test_unsupported_or_excessive_commands_stop(self):
        scene = self.scene()
        for v, w in ((math.nan, 0.), (.2, 0.), (0., .6)):
            self.assertFalse(scene.safe(v, w))
        with self.assertRaises(ValueError):
            self.scene([math.nan]*721)

    def test_arc_geometry(self):
        np.testing.assert_allclose(arc_pose(.06, 0., 2.), [.12, 0., 0.])
        np.testing.assert_allclose(arc_pose(.06, .2, 2.), [.3*math.sin(.4), .3*(1-math.cos(.4)), .4])

    def test_sweep_checks_later_segments_in_original_observation_frame(self):
        scene = self.scene()
        self.assertTrue(scene.safe(.06, 0., start=[.1, 0., .1]))
        self.assertFalse(scene.safe(.06, 0., start=[3., 0., 0.]))
        self.assertFalse(scene.safe(.06, 0., start=[float('nan'), 0., 0.]))

    def test_open_space_keeps_direct_motion_without_escape_sequence(self):
        scene = self.scene()
        self.assertEqual(scene.choose(0.)[:2], (.06, 0.))
        self.assertIsNone(scene.recovery_path)

    def recovery_fixture(self):
        guard = ObservedFootprintGuard.__new__(ObservedFootprintGuard)
        guard.recovery = ([(.06, 0., np.array([.09, 0., 0.])),
                           (0., .24, np.array([.09, 0., .36]))], 10.)
        scene = NS(pose=np.zeros(3), safe=Mock(return_value=True), recovery_path=None,
                   choose=Mock(return_value=(0., 0., 'NO_OBSERVED_SWEEP')))
        return guard, scene

    def test_recovery_keeps_selected_subgoal_until_measured_reached(self):
        guard, scene = self.recovery_fixture()
        self.assertEqual(guard.recovery_command(scene, -.4, 11.)[:2], (.06, 0.))
        scene.choose.assert_not_called()
        scene.pose = np.array([.09, 0., 0.])
        self.assertEqual(guard.recovery_command(scene, -.4, 11.5)[:2], (0., .24))
        self.assertEqual(len(guard.recovery[0]), 1)

    def test_expired_recovery_replans_without_claiming_completion(self):
        guard, scene = self.recovery_fixture()
        guard.recovery_command(scene, 0., 16.)
        self.assertIsNone(guard.recovery)
        scene.choose.assert_called_once()

    def test_blocked_recovery_stops_and_rechecks_same_subgoal(self):
        guard, scene = self.recovery_fixture()
        original = guard.recovery
        scene.safe.return_value = False
        self.assertEqual(guard.recovery_command(scene,0.,11.),(0.,0.,'OBSERVED_RECOVERY_BLOCKED'))
        self.assertIs(guard.recovery,original)
        scene.choose.assert_not_called()
        self.assertIsNone(getattr(guard,'last_recovery_progress_time',None))
        scene.safe.return_value = True
        self.assertEqual(guard.recovery_command(scene,0.,11.1)[:2],(.06,0.))
        self.assertEqual(scene.safe.call_count,2)

    def test_continuously_blocked_plan_expires_without_extending_deadline(self):
        guard, scene = self.recovery_fixture()
        scene.safe.return_value = False
        for now in (11.,12.,14.9):
            self.assertEqual(guard.recovery_command(scene,0.,now)[:2],(0.,0.))
            self.assertEqual(guard.recovery[1],10.)
        guard.recovery_command(scene,0.,15.1)
        self.assertIsNone(guard.recovery)
        scene.choose.assert_called_once()

    def test_subgoal_command_prediction_ends_at_remaining_translation(self):
        guard, scene = self.recovery_fixture()
        scene.pose = np.array([.06, 0., 0.])
        v, w, mode = guard.recovery_command(scene, 0., 11.)
        self.assertAlmostEqual(v, .02)
        self.assertEqual(w, 0.)
        self.assertEqual(mode, 'OBSERVED_RECOVERY_SUBGOAL')
        scene.safe.assert_called_once_with(v, w)

    def test_subgoal_command_prediction_ends_at_remaining_rotation(self):
        guard, scene = self.recovery_fixture()
        guard.recovery = ([(0., .24, np.array([0., 0., .36]))], 10.)
        scene.pose = np.array([0., 0., .30])
        v, w, mode = guard.recovery_command(scene, 0., 11.)
        self.assertEqual(v, 0.)
        self.assertAlmostEqual(w, .04)
        self.assertEqual(mode, 'OBSERVED_RECOVERY_SUBGOAL')

    def test_translation_behind_subgoal_does_not_keep_advancing(self):
        guard, scene = self.recovery_fixture()
        scene.pose = np.array([.13, 0., 0.])
        guard.recovery_command(scene, 0., 11.)
        scene.choose.assert_called_once()

    def test_arc_endpoint_advances_with_current_scene_safety_check(self):
        guard,scene=self.recovery_fixture()
        first=arc_pose(.06,.24,1.5)
        guard.recovery=([(.06,.24,first),(.06,0.,first+np.array([.09,0.,0.]))],10.)
        scene.pose=first+np.array([-.025,0.,-.02])
        v,w,_=guard.recovery_command(scene,.2,11.)
        self.assertEqual(len(guard.recovery[0]),1)
        self.assertGreater(v,0.)
        self.assertEqual(w,0.)
        scene.safe.assert_called_once_with(v,w)

    def test_arc_endpoint_tolerance_never_approves_a_blocked_next_segment(self):
        guard,scene=self.recovery_fixture()
        first=arc_pose(.06,.24,1.5)
        guard.recovery=([(.06,.24,first),(.06,0.,first+np.array([.09,0.,0.]))],10.)
        scene.pose=first+np.array([-.025,0.,-.02])
        scene.safe.return_value=False
        self.assertEqual(guard.recovery_command(scene,0.,11.)[:2],(0.,0.))
        self.assertIsNotNone(guard.recovery)
        scene.choose.assert_not_called()

    def test_missing_scene_preserves_bounded_subgoal_but_stops(self):
        guard, scene = self.recovery_fixture()
        self.assertEqual(guard.recovery_command(None, 0., 11.)[:2], (0., 0.))
        self.assertIsNotNone(guard.recovery)
        self.assertEqual(guard.recovery_command(scene,0.,11.1)[:2],(.06,0.))
        scene.safe.assert_called_once()
        scene.choose.assert_not_called()

    def test_long_sensor_gap_discards_expired_subgoal_without_motion(self):
        guard, scene = self.recovery_fixture()
        self.assertEqual(guard.recovery_command(None,0.,16.)[:2],(0.,0.))
        self.assertIsNone(guard.recovery)

    def test_recovery_observed_subgoal_is_transformed_to_map(self):
        guard, scene = self.recovery_fixture()
        guard.recovery = None
        scene.pose = np.array([2., 3., math.pi/2])
        scene.recovery_path = [(-.06, 0., np.array([-.09, 0., 0.]))]
        scene.choose.return_value = (-.06, 0., 'OBSERVED_MULTI_STEP_SWEEP')
        guard.recovery_command(scene, .2, 10.)
        np.testing.assert_allclose(guard.recovery[0][0][2], [2., 2.91, math.pi/2])

    def test_cross_track_missed_subgoal_replans_instead_of_asymptotic_crawl(self):
        guard,scene=self.recovery_fixture()
        scene.pose=np.array([.078,.025,0.])
        command=guard.recovery_command(scene,0.,11.)
        scene.choose.assert_called_once()
        self.assertEqual(command,(0.,0.,'NO_OBSERVED_SWEEP'))
        self.assertIsNone(guard.recovery)

    def test_local_recovery_tracks_own_odom_not_slam_map_jump(self):
        guard,scene=self.recovery_fixture()
        guard.recovery_frame='odom'
        scene.odom_pose=np.array([.06,0.,0.])
        scene.pose=np.array([3.,-2.,.5])
        v,w,mode=guard.recovery_command(scene,0.,11.)
        self.assertAlmostEqual(v,.02)
        self.assertEqual(w,0.)
        self.assertEqual(mode,'OBSERVED_RECOVERY_SUBGOAL')
        scene.safe.assert_called_once_with(v,w)

    def test_odom_subgoals_are_not_reinterpreted_as_map_when_tf_is_missing(self):
        guard,scene=self.recovery_fixture()
        guard.recovery_frame='odom'
        guard.recovery_command(scene,0.,11.)
        scene.choose.assert_called_once()
        self.assertIsNone(guard.recovery)

    def test_new_recovery_plan_is_stored_in_observed_odom_frame(self):
        guard,scene=self.recovery_fixture()
        guard.recovery=None
        scene.pose=np.array([20.,30.,0.])
        scene.odom_pose=np.array([2.,3.,math.pi/2])
        scene.recovery_path=[(-.06,0.,np.array([-.09,0.,0.]))]
        scene.choose.return_value=(-.06,0.,'OBSERVED_MULTI_STEP_SWEEP')
        guard.recovery_command(scene,.2,10.)
        np.testing.assert_allclose(guard.recovery[0][0][2],[2.,2.91,math.pi/2])
        self.assertEqual(guard.recovery_frame,'odom')

    def test_invalid_tracking_pose_stops_without_executing_old_plan(self):
        for pose in ([math.nan,0.,0.],[0.,0.]):
            guard,scene=self.recovery_fixture()
            scene.odom_pose=pose
            self.assertEqual(guard.recovery_command(scene,0.,11.)[:2],(0.,0.))
            scene.safe.assert_not_called()
            scene.choose.assert_not_called()
            self.assertIsNone(guard.recovery)

    def test_measured_escape_subgoal_completion_credits_only_own_odom_motion(self):
        guard,scene=self.recovery_fixture()
        guard.recovery_frame='odom'
        guard.recovery_segment_start=np.zeros(3)
        scene.odom_pose=np.array([.08,0.,0.])
        guard.recovery_command(scene,0.,11.5)
        self.assertEqual(guard.last_recovery_progress_time,11.5)
        np.testing.assert_allclose(guard.recovery_segment_start,scene.odom_pose)

    def test_map_only_subgoal_or_unmeasured_initial_proximity_cannot_earn_credit(self):
        for kind in ('map','no_motion','jump'):
            guard,scene=self.recovery_fixture()
            scene.pose=np.array([.09,0.,0.])
            guard.recovery_segment_start=np.zeros(3)
            if kind!='map':
                guard.recovery_frame='odom'
                scene.odom_pose=scene.pose.copy()
            if kind=='no_motion':guard.recovery_segment_start=scene.pose.copy()
            if kind=='jump':guard.recovery_segment_start=np.array([-2.,0.,0.])
            guard.recovery_command(scene,0.,11.5)
            self.assertIsNone(getattr(guard,'last_recovery_progress_time',None))

    def test_new_job_reset_removes_old_progress_credit(self):
        guard,scene=self.recovery_fixture()
        guard.last_recovery_progress_time=10.
        guard.recovery_segment_start=np.zeros(3)
        guard.reset_recovery()
        self.assertIsNone(guard.last_recovery_progress_time)
        self.assertIsNone(guard.recovery_segment_start)

    def test_extended_planning_keeps_robot_stopped_without_blocking_callback(self):
        guard,scene=self.recovery_fixture()
        guard.recovery=None
        scene.choose.return_value=(0.,0.,'NO_OBSERVED_SWEEP')
        scene.extended_recovery_path=Mock()
        pending=Future()
        guard.recovery_executor=Mock()
        guard.recovery_executor.submit.return_value=pending
        self.assertEqual(guard.recovery_command(scene,0.,10.),(0.,0.,'PLANNING_OBSERVED_ESCAPE'))
        self.assertEqual(guard.recovery_command(scene,0.,10.1),(0.,0.,'PLANNING_OBSERVED_ESCAPE'))
        guard.recovery_executor.submit.assert_called_once()

    def test_safe_fine_progress_is_not_replaced_by_a_stopped_long_search(self):
        for speed in (.015,.03):
            guard,scene=self.recovery_fixture()
            guard.recovery=None
            scene.choose.return_value=(speed,0.,'OBSERVED_FOOTPRINT_SWEEP')
            scene.extended_recovery_path=Mock()
            guard.recovery_executor=Mock()
            self.assertEqual(guard.recovery_command(scene,0.,10.),
                             (speed,0.,'OBSERVED_FOOTPRINT_SWEEP'))
            guard.recovery_executor.submit.assert_not_called()

    def test_safe_fine_progress_does_not_wait_for_a_discarded_search(self):
        guard,scene=self.recovery_fixture()
        guard.recovery=None
        guard.recovery_task=Future()
        scene.choose.return_value=(.015,0.,'OBSERVED_FOOTPRINT_SWEEP')
        scene.extended_recovery_path=Mock()
        self.assertEqual(guard.recovery_command(scene,0.,10.),
                         (.015,0.,'OBSERVED_FOOTPRINT_SWEEP'))
        self.assertFalse(guard.recovery_task.done())

    def pending_plan(self):
        guard,scene=self.recovery_fixture()
        guard.recovery=None
        future=Future()
        future.set_result([(-.06,0.,np.array([-.09,0.,0.]))])
        guard.pending_recovery=(future,np.zeros(3),'map',10.)
        return guard,scene

    def test_finished_plan_is_rechecked_against_current_scan(self):
        guard,scene=self.pending_plan()
        self.assertEqual(guard.recovery_command(scene,0.,11.)[:2],(-.06,0.))
        scene.safe.assert_called_once_with(-.06,0.)

    def test_new_obstacle_stops_finished_plan_until_fresh_clearance_or_expiry(self):
        guard,scene=self.pending_plan()
        scene.safe.return_value=False
        self.assertEqual(guard.recovery_command(scene,0.,11.)[:2],(0.,0.))
        scene.choose.assert_not_called()
        self.assertIsNotNone(guard.recovery)

    def test_moved_robot_expired_plan_or_changed_frame_discards_proposal(self):
        for kind in ('moved','turned','frame'):
            guard,scene=self.pending_plan()
            if kind=='moved':scene.pose=np.array([.05,0.,0.])
            if kind=='turned':scene.pose=np.array([0.,0.,.07])
            if kind=='frame':scene.odom_pose=np.zeros(3)
            guard.recovery_command(scene,0.,11.)
            scene.safe.assert_not_called()
            scene.choose.assert_called_once()
            self.assertIsNone(guard.recovery)

    def test_expired_plan_is_discarded_without_executing_motion(self):
        guard,scene=self.pending_plan()
        self.assertEqual(guard.recovery_command(scene,0.,26.),(0.,0.,'EXPIRED_ESCAPE_PLAN'))
        scene.safe.assert_not_called()
        scene.choose.assert_not_called()
        self.assertIsNone(guard.pending_recovery)

    def test_transient_missing_scan_stops_but_keeps_pending_computation(self):
        guard,scene=self.pending_plan()
        pending=guard.pending_recovery
        self.assertEqual(guard.recovery_command(None,0.,11.)[:2],(0.,0.))
        self.assertIs(guard.pending_recovery,pending)
        self.assertEqual(guard.recovery_command(scene,0.,11.1)[:2],(-.06,0.))
        scene.safe.assert_called_once_with(-.06,0.)

    def test_reset_between_jobs_discards_proposal(self):
        guard,scene=self.pending_plan()
        guard.reset_recovery()
        self.assertIsNone(guard.pending_recovery)
        self.assertIsNone(guard.recovery)

    def test_unfinished_discarded_task_cannot_queue_another_search(self):
        guard,scene=self.recovery_fixture()
        guard.recovery=None
        guard.recovery_task=Future()
        guard.recovery_executor=Mock()
        scene.choose.return_value=(0.,0.,'NO_OBSERVED_PROGRESS')
        scene.extended_recovery_path=Mock()
        self.assertEqual(guard.recovery_command(scene,0.,11.),(0.,0.,'DRAINING_OBSERVED_ESCAPE'))
        guard.recovery_executor.submit.assert_not_called()

    def test_extended_search_keeps_original_safety_predicate(self):
        scene=self.scene()
        scene.safe=Mock(return_value=False)
        self.assertIsNone(scene.extended_recovery_path(0.))

    def history(self, stamps=(9.6,9.8), ranges=None):
        scan=self.scene().scan
        if ranges is not None:
            scan.ranges=ranges
        return [(t,np.zeros(3),np.array([.25,0.,0.]),scan) for t in stamps]

    def test_two_recent_measured_views_can_support_current_blind_rays(self):
        ranges=[3.]*721
        ranges[:90]=[math.nan]*90
        ranges[-90:]=[math.nan]*90
        scene=self.scene(ranges)
        self.assertFalse(scene.safe(-.06,0.))
        scene.free_history=self.history()
        scene.odom_pose=np.zeros(3);scene.measurement_stamp=10.
        self.assertTrue(scene.safe(-.06,0.))

    def test_history_never_erases_current_obstacle_or_its_shadow(self):
        scene=self.scene()
        scene.scan.ranges[359:362]=[.5,math.nan,3.]
        scene=self.scene(scene.scan.ranges)
        scene.free_history=self.history()
        scene.odom_pose=np.zeros(3);scene.measurement_stamp=10.
        self.assertFalse(scene.observed_free(np.array([[1.,0.]])).all())

    def test_old_future_duplicate_or_single_views_cannot_certify_blind_space(self):
        points=np.array([[-.4,0.]])
        for stamps in ((7.,7.2),(10.1,10.3),(9.8,9.8),(9.8,), (9.8,9.85)):
            self.assertFalse(historical_ray_support(points,np.zeros(3),10.,self.history(stamps)).any())

    def test_history_requires_measured_rays_not_unknowns(self):
        history=self.history(ranges=[math.nan]*721)
        self.assertFalse(historical_ray_support(np.array([[-.4,0.]]),np.zeros(3),10.,history).any())

    def test_history_wheel_transform_prevents_treating_old_ray_as_current(self):
        history=self.history(ranges=[1.]*721)
        points=np.array([[.4,0.]])
        self.assertTrue(historical_ray_support(points,np.zeros(3),10.,history).all())
        self.assertFalse(historical_ray_support(points,np.array([1.,0.,0.]),10.,history).any())

    def test_fine_observed_step_does_not_relax_sweep_safety(self):
        scene=self.scene()
        scene.safe=Mock(side_effect=lambda v,w: 0.<v<=.03 and abs(w)<1e-8)
        v,w,_=scene.choose(0.)
        self.assertEqual((v,w),(.03,0.))

    def test_no_safe_fine_step_remains_stopped(self):
        scene=self.scene()
        scene.safe=Mock(return_value=False)
        self.assertEqual(scene.choose(.5)[:2],(0.,0.))

    def test_safe_return_turn_precedes_forward_progress_toward_jamb(self):
        scene=self.scene()
        for error in (.8,-.8):
            v,w,mode=scene.choose(error)
            self.assertEqual(v,0.)
            self.assertEqual(w,math.copysign(.24,error))
            self.assertEqual(mode,'OBSERVED_RETURN_HEADING')

    def test_return_turn_is_limited_to_remaining_observed_heading(self):
        scene=self.scene()
        v,w,_=scene.choose(.21)
        self.assertEqual(v,0.)
        self.assertAlmostEqual(w*1.5,.21)

    def test_return_turn_does_not_bypass_unknown_sweep(self):
        scene=self.scene()
        scene.safe=Mock(side_effect=lambda v,w: 0.<v<=.03 and w==0.)
        v,w,mode=scene.choose(.8)
        self.assertEqual((v,w),(.03,0.))
        self.assertNotEqual(mode,'OBSERVED_RETURN_HEADING')

    def test_small_return_rotation_uses_unchanged_full_horizon_guard(self):
        scene=self.scene()
        scene.safe=Mock(side_effect=lambda v,w: v==0. and 0.<w<=.04)
        self.assertEqual(scene.choose(.8)[:2],(0.,.04))
        scene.safe.assert_any_call(0.,.24)
        scene.safe.assert_any_call(0.,.04)

    def wall_scene(self, wall_x):
        angles=np.linspace(-math.pi,math.pi,721)
        ranges=np.full(721,3.)
        forward=np.cos(angles)>0.
        ranges[forward]=np.minimum(3.,(wall_x-.25)/np.cos(angles[forward]))
        return self.scene(ranges.tolist())

    def test_expanded_body_margin_is_not_added_twice_to_rays(self):
        scene=self.wall_scene(.43)
        self.assertTrue(scene.safe(.03,0.))
        self.assertGreaterEqual(scene.clearance(arc_pose(.03,0.,1.5)),.05)
        self.assertAlmostEqual(scene.start_hi[0],.362)

    def test_original_five_centimeter_collision_margin_still_blocks(self):
        scene=self.wall_scene(.395)
        self.assertFalse(scene.safe(.03,0.))
        self.assertLess(scene.clearance(arc_pose(.03,0.,1.5)),.05)

    def test_padded_ray_query_never_accepts_unknown_or_obstacle_shadow(self):
        scene=self.scene()
        scene.scan.ranges[359:362]=[.5,math.nan,3.]
        scene=self.scene(scene.scan.ranges)
        self.assertFalse(scene.observed_free(np.array([[1.,0.]]),padded=True).any())
        scene=self.scene([3.]*359+[math.nan]*3+[3.]*359)
        self.assertFalse(scene.observed_free(np.array([[.4,0.]]),padded=True).any())


if __name__ == '__main__':
    unittest.main()
