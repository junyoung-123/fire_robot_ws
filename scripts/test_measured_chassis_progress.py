import math
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from fire_robot_manipulation.contact_feedback import (
    observed_probe_progress, observed_probe_motion, measured_contact_endpoint,
    measured_tangential_relaxation, bounded_probe_pulse_scale)


class DelayedProbePulseTests(unittest.TestCase):
    def test_alignment_clock_budget_preserves_sim_limit_and_stall_detection(self):
        from fire_robot_manipulation.contact_feedback import SimMotionDeadline
        source=Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation/fire_robot_manipulation/mission_contact_adapter.py'
        tree=ast.parse(source.read_text())
        opening=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='open')
        deadline=next(node for node in ast.walk(opening) if isinstance(node,ast.Call)
            and ast.unparse(node.func)=='SimMotionDeadline')
        self.assertEqual(ast.literal_eval(deadline.args[2]),100.)
        self.assertEqual({kw.arg:ast.literal_eval(kw.value) for kw in deadline.keywords},
            {'stall_limit':3.,'wall_limit':900.})
        clock=SimMotionDeadline(0.,0.,100.,stall_limit=3.,wall_limit=900.)
        for wall in range(1,501):
            self.assertIsNone(clock.failure(wall*.15,float(wall)))
        self.assertEqual(clock.failure(75.,503.),'Simulation clock stalled')
        self.assertEqual(SimMotionDeadline(0.,0.,100.).failure(100.,1.),
            'Simulation motion deadline exceeded')

    def test_recorded_short_turn_retains_latency_headroom_without_relaxing_guard(self):
        duration=.0875
        for progress in (0.,.5,.9,.99):
            scale=bounded_probe_pulse_scale(progress,duration,.05,.04,.05)
            self.assertLessEqual(scale,.461* (1.-progress))
            self.assertLessEqual(scale*(.05+.04+2.*.05),(1.-progress)*duration+1.e-12)
        with self.assertRaisesRegex(ValueError,'opposes or exceeds'):
            observed_probe_motion([5.636233173973906,-1.5452857585139432,-1.8039684852753417],
                [5.636243044960397,-1.5452436300471255,-1.7979340544657822],
                [.6967,.079,.738],[1.,0.,0.],-.00038835188938413784,
                .05416014802651667,duration,.008)

    def test_scaling_is_common_to_both_velocity_components(self):
        for duration in (.0875,.175,.35):
            scale=bounded_probe_pulse_scale(.6,duration,.05,.02,.05)
            self.assertGreater(scale,0.)
            self.assertLessEqual(scale,.4)
            self.assertAlmostEqual((.02*scale)/(-.08*scale),-.25)

    def test_completed_or_invalid_feedback_never_authorizes_extra_motion(self):
        self.assertEqual(bounded_probe_pulse_scale(1.,.1,.05,0.,.05),0.)
        for age,period in ((-.001,.05),(.151,.05),(0.,0.),(0.,.151),(0.,float('nan'))):
            with self.assertRaises(ValueError):
                bounded_probe_pulse_scale(.5,.1,.05,age,period)

    def test_fixed_urdf_transform_cache_is_numerically_identical(self):
        from fire_robot_manipulation.piper_actual_kinematics import (
            PiperActualKinematics,PIPER_JOINT_LIMITS,_JOINT_ORIGINS,_transform,_rot_z)
        arm=PiperActualKinematics()
        rng=np.random.default_rng(150)
        for q in rng.uniform(PIPER_JOINT_LIMITS[:,0],PIPER_JOINT_LIMITS[:,1],size=(100,6)):
            expected=_transform(arm.mount_xyz)
            for (xyz,rpy),angle in zip(_JOINT_ORIGINS,q):
                joint=np.eye(4); joint[:3,:3]=_rot_z(float(angle))
                expected=expected@_transform(xyz,rpy)@joint
            point,rotation=arm.forward(q)
            np.testing.assert_array_equal(point,(expected@np.array([0.,0.,arm.tool_contact_offset_m,1.]))[:3])
            np.testing.assert_array_equal(rotation,expected[:3,:3])


class MeasuredContactEndpointTests(unittest.TestCase):
    def test_stopped_base_cannot_supply_requested_push(self):
        point=np.array([.55,.05,.74])
        result,advance=measured_contact_endpoint([0.,0.,0.],[0.,0.,0.],point,[1.,0.,0.],.008,.008)
        np.testing.assert_allclose(result,point)
        self.assertEqual(advance,0.)

    def test_sideways_motion_is_compensated_exactly_not_projected_away(self):
        point=np.array([.55,.05,.74])
        result,advance=measured_contact_endpoint([0.,0.,0.],[.003,.002,0.],point,[1.,0.,0.],.004,.008)
        np.testing.assert_allclose(result,[.55,.048,.74])
        self.assertAlmostEqual(advance,.003)

    def test_reverse_with_no_normal_push_keeps_world_contact_fixed(self):
        point=np.array([.55,.05,.74]); pose=[-.001,0.,.005]
        result,advance=measured_contact_endpoint([0.,0.,0.],pose,point,[1.,0.,0.],0.,.008)
        c,s=math.cos(pose[2]),math.sin(pose[2])
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        np.testing.assert_allclose(rotation@result+[-.001,0.,0.],point,atol=1.e-10)
        self.assertEqual(advance,0.)

    def test_excessive_measured_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'bounded contact'):
            measured_contact_endpoint([0.,0.,0.],[0.,.03,0.],[.55,.05,.74],[1.,0.,0.],0.,.008)

    def test_world_origin_and_yaw_do_not_change_exact_compensation(self):
        point=[.55,.05,.74]; local=np.array([.003,.002,.005])
        heading=1.1; c,s=math.cos(heading),math.sin(heading)
        rotation=np.array([[c,-s],[s,c]])
        initial=np.array([22.,-17.,heading])
        moved=np.r_[initial[:2]+rotation@local[:2],heading+local[2]]
        a,_=measured_contact_endpoint([0.,0.,0.],local,point,[1.,0.,0.],.004,.008)
        b,_=measured_contact_endpoint(initial,moved,point,[1.,0.,0.],.004,.008)
        np.testing.assert_allclose(a,b,atol=1.e-10)


class TangentialContactRelaxationTests(unittest.TestCase):
    def test_bounded_relaxation_cannot_add_normal_push_or_vertical_press(self):
        normal=np.array([math.cos(-.6),math.sin(-.6),0.])
        tangent=np.array([-normal[1],normal[0],0.]); target=np.array([.58,-.14,.73])
        result=measured_tangential_relaxation(target,target+.02*tangent,normal,.04,.008,.015)
        np.testing.assert_allclose(result-target,.008*tangent)
        self.assertAlmostEqual((result-target)@normal,0.)
        self.assertEqual(result[2],target[2])

    def test_normal_jam_is_not_hidden_by_reanchoring(self):
        for actual in ([.6,0.,.74],[.58,0.,.76]):
            with self.assertRaisesRegex(ValueError,'normal or vertical'):
                measured_tangential_relaxation([.58,0.,.74],actual,[1.,0.,0.],.04,.008,.015)

    def test_excessive_contact_displacement_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'excessive'):
            measured_tangential_relaxation([.58,0.,.74],[.58,.05,.74],[1.,0.,0.],.04,.008,.015)


class MeasuredChassisProgressTests(unittest.TestCase):
    def probe_method(self):
        source = Path(__file__).resolve().parents[1] / 'src/fire_robot_manipulation/fire_robot_manipulation/preserved_contact_policy.py'
        tree = ast.parse(source.read_text())
        return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == '_feedback_probe_contact_motion')

    def probe_continues(self, coordinated, fraction, elapsed, ros_ok=True):
        loop = next(node for node in ast.walk(self.probe_method()) if isinstance(node, ast.While))
        expression = compile(ast.Expression(loop.test), '<actual_probe_loop>', 'eval')
        return eval(expression, {
            'rclpy': SimpleNamespace(ok=lambda: ros_ok),
            'self': SimpleNamespace(_sim_time_sec=lambda: elapsed),
            'coordinated': coordinated, 'chassis_fraction': fraction, 'start': 0., 'duration': .1})

    def test_solver_latency_does_not_finish_incomplete_measured_probe(self):
        for fraction in (0., .4, .99):
            self.assertTrue(self.probe_continues(object(), fraction, .6))

    def test_measured_completion_stops_even_before_nominal_duration(self):
        self.assertFalse(self.probe_continues(object(), 1., .05))
        self.assertFalse(self.probe_continues(object(), 1., .6))
        self.assertFalse(self.probe_continues(object(), .4, .05, ros_ok=False))

    def test_uncoordinated_probe_keeps_original_duration_limit(self):
        self.assertTrue(self.probe_continues(None, 0., .05))
        self.assertFalse(self.probe_continues(None, 0., .1))
        self.assertFalse(self.probe_continues(None, 0., .6))

    def test_measured_probe_keeps_deadline_and_displacement_guards(self):
        method = self.probe_method()
        deadline = next(node for node in ast.walk(method) if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == 'SimMotionDeadline')
        self.assertEqual(ast.unparse(deadline.args[2]), 'duration + 1.0')
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in deadline.keywords},
                         {'stall_limit': 2., 'wall_limit': 10.})
        calls = {ast.unparse(node.func) for node in ast.walk(method) if isinstance(node, ast.Call)}
        for guard in ('deadline.failure', 'observed_probe_motion', 'measured_contact_endpoint',
                      'self._feedback_scene_command_safe', 'self._feedback_fresh'):
            self.assertIn(guard, calls)
        stop = next(node for node in ast.walk(method) if isinstance(node, ast.If)
                    and ast.unparse(node.test).startswith('chassis_fraction >= 1.0'))
        self.assertTrue(any(isinstance(node, ast.Break) for node in stop.body))

    def test_completion_tolerance_is_the_existing_measured_rest_tolerance(self):
        stop = next(node for node in ast.walk(self.probe_method()) if isinstance(node, ast.If)
                    and ast.unparse(node.test).startswith('chassis_fraction >= 1.0'))
        expression = compile(ast.Expression(stop.test), '<actual_measured_completion>', 'eval')
        env = {'np': np, 'chassis_fraction': .90, 'duration': .2, 'tip_speed': .04,
               'cfg': {'follow_step_m': .008}, 'goal': np.zeros(3), 'origin': np.zeros(3)}
        self.assertFalse(eval(expression, env))
        env['chassis_fraction'] = .96
        self.assertTrue(eval(expression, env))
        env['chassis_fraction'] = .4
        self.assertFalse(eval(expression, env))

    def test_pulse_taper_preserves_the_checked_turning_radius(self):
        calls = [node for node in ast.walk(self.probe_method()) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == 'self._publish_sim_cmd_vel']
        call = next(node for node in calls if ast.unparse(node.args[0]) == 'speed * pulse_scale')
        self.assertEqual(ast.unparse(call.args[1]), 'angular_z * pulse_scale')
        assignment = next(node for node in ast.walk(self.probe_method()) if isinstance(node, ast.Assign)
                          and any(ast.unparse(target) == 'pulse_scale' for target in node.targets))
        expression = compile(ast.Expression(assignment.value), '<actual_taper>', 'eval')
        self.assertAlmostEqual(eval(expression, {'chassis_fraction': .8}), .2)

    def test_expensive_checks_run_with_chassis_command_stopped(self):
        loop = next(node for node in ast.walk(self.probe_method()) if isinstance(node, ast.While))
        first = loop.body[0]
        self.assertEqual(ast.unparse(first.test), 'coordinated is not None')
        self.assertEqual(ast.unparse(first.body[0]), 'self._publish_sim_cmd_vel(0.0, 0.0)')

    def test_active_pulse_is_short_and_always_stops_before_next_solver_pass(self):
        method = self.probe_method()
        pulse = next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                     and any(ast.unparse(target) == 'pulse_duration' for target in node.targets))
        expression = compile(ast.Expression(pulse.value), '<actual_pulse_limit>', 'eval')
        env = {'np': np, 'duration': .2, 'pulse_scale': 1., 'cfg': {'follow_step_m': .008},
               'tip_speed': .04}
        self.assertAlmostEqual(eval(expression, env), .05)
        env['duration'] = .02
        self.assertAlmostEqual(eval(expression, env), .02)
        env.update(duration=.2, tip_speed=.10)
        self.assertAlmostEqual(eval(expression, env), .02)
        stopped = next(node for node in ast.walk(method) if isinstance(node, ast.Try)
                       and any(isinstance(child, ast.While) and 'pulse_start' in ast.unparse(child.test)
                               for child in node.body))
        self.assertEqual(ast.unparse(stopped.finalbody[0]), 'self._publish_sim_cmd_vel(0.0, 0.0)')

    def test_committed_target_uses_its_calculation_pose_timestamp(self):
        method = self.probe_method()
        assignments = [node for node in ast.walk(method) if isinstance(node, ast.Assign)]
        assignment = next(node for node in assignments if any(
            ast.unparse(target) == 'point.header.stamp' for target in node.targets))
        self.assertEqual(ast.unparse(assignment.value), 'Time(seconds=last_target_stamp).to_msg()')
        sources = {ast.unparse(node.value) for node in assignments if any(
            ast.unparse(target) == 'last_target_stamp' for target in node.targets)}
        self.assertEqual(sources, {'initial_pose_stamp', 'pose_stamp'})

    def progress(self,current,speed=.02,yaw_rate=-.04,initial=(0.,0.,0.)):
        return observed_probe_progress(initial,current,[.56,.08,.74],
            [math.cos(-.4),math.sin(-.4),0.],speed,yaw_rate,.3,.008)

    def pose(self,fraction,speed=.02,yaw_rate=-.04):
        angle=yaw_rate*.3*fraction
        if abs(yaw_rate)<1.e-8:
            return [speed*.3*fraction,0.,0.]
        return [speed/yaw_rate*math.sin(angle),
                speed/yaw_rate*(1.-math.cos(angle)),angle]

    def test_blocked_chassis_never_advances_arm(self):
        self.assertEqual(self.progress([0.,0.,0.])[0],0.)

    def test_progress_matches_measured_motion_not_wall_time(self):
        for fraction in (0.,.1,.5,.9,1.):
            observed,residual=self.progress(self.pose(fraction))
            self.assertAlmostEqual(observed,fraction,delta=.001)
            self.assertLess(residual,1.e-5)

    def test_normal_supply_caps_progress_when_wheels_do_not_translate(self):
        # Here turning without the commanded translation would close the door.
        observed,_=observed_probe_progress([0.,0.,0.],[0.,0.,-.012],[.56,.08,.74],
            [math.cos(.4),math.sin(.4),0.],.02,-.04,.3,.008)
        self.assertEqual(observed,0.)

    def test_completed_steering_stops_chassis_without_inventing_normal_arm_push(self):
        advance,residual,travel=observed_probe_motion([0.,0.,0.],[0.,0.,-.012],
            [.56,.08,.74],[math.cos(.4),math.sin(.4),0.],.02,-.04,.3,.008)
        self.assertEqual(advance,0.)
        self.assertEqual(travel,1.)
        self.assertLess(residual,.008)

    def test_completed_translation_stops_segment_when_turning_lags(self):
        pose = self.pose(1.)
        pose[2] = 0.
        advance,residual,travel=observed_probe_motion([0.,0.,0.],pose,
            [.56,.08,.74],[1.,0.,0.],.02,-.04,.3,.008)
        self.assertEqual(travel,1.)
        self.assertLess(residual,.008)
        self.assertLessEqual(advance,1.)

    def test_split_progress_keeps_original_reverse_and_overrun_rejections(self):
        for pose in (self.pose(-.5),self.pose(2.)):
            with self.assertRaises(ValueError):
                observed_probe_motion([0.,0.,0.],pose,[.56,.08,.74],
                    [math.cos(-.4),math.sin(-.4),0.],.02,-.04,.3,.008)

    def test_world_origin_and_heading_do_not_change_progress(self):
        local=np.array(self.pose(.5))
        heading=1.1
        rotation=np.array([[math.cos(heading),-math.sin(heading)],
                           [math.sin(heading),math.cos(heading)]])
        current=np.r_[np.array([22.,-17.])+rotation@local[:2],heading+local[2]]
        self.assertAlmostEqual(self.progress(current,initial=(22.,-17.,heading))[0],.5,delta=.001)

    def test_straight_reverse_is_measured_without_fake_forward_push(self):
        observed,_=self.progress(self.pose(.5,speed=-.02,yaw_rate=0.),speed=-.02,yaw_rate=0.)
        self.assertAlmostEqual(observed,.5)

    def test_unexpected_sideways_motion_stops(self):
        with self.assertRaises(ValueError):
            self.progress([0.,.03,0.])

    def test_backward_or_excessive_motion_stops(self):
        for current in (self.pose(-.5),self.pose(2.)):
            with self.assertRaises(ValueError):
                self.progress(current)

    def test_uncommanded_motion_and_nonfinite_pose_stop(self):
        for current in ([.02,0.,0.],[float('nan'),0.,0.]):
            with self.assertRaises(ValueError):
                self.progress(current,speed=0.,yaw_rate=0.)


if __name__=='__main__':
    unittest.main()
