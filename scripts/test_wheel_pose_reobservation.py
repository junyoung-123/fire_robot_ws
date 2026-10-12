from collections import deque
import ast
import inspect
import textwrap
import unittest
from unittest.mock import Mock,patch
import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from fire_robot_manipulation.contact_feedback import SceneObservationPending


class WheelPoseReobservationTests(unittest.TestCase):
    def fixture(self,stamp=1.,received=10.):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        clock=[1.,10.]
        node._contact_wheel_samples=deque([(stamp,received,np.array([.1,.2,.3]))])
        node._sim_time_sec=lambda:clock[0]
        node._feedback_measurement_pause_generation=0
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        return node,clock

    def call(self,node,clock,sleep):
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True),\
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',side_effect=lambda:clock[1]),\
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.sleep',side_effect=sleep):
            return node._feedback_wait_wheel_pose_sample()

    def test_fresh_sample_returns_without_chassis_command(self):
        node,clock=self.fixture()
        result=self.call(node,clock,Mock())
        self.assertEqual(result[0],1.)
        node._publish_sim_cmd_vel.assert_not_called()
        self.assertEqual(node._feedback_measurement_pause_generation,0)

    def test_future_packet_waits_for_clock_without_retimestamping(self):
        node,clock=self.fixture(stamp=1.05)
        def tick(_):
            clock[0]=1.06; clock[1]=10.1
        result=self.call(node,clock,tick)
        self.assertEqual(result[0],1.05)
        self.assertEqual(node._contact_wheel_samples[0][0],1.05)
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)
        self.assertEqual(node._feedback_measurement_pause_generation,1)

    def test_late_receipt_cannot_be_made_fresh_but_new_packet_can_recover(self):
        node,clock=self.fixture(received=8.)
        def receive(_):
            clock[0]=1.1; clock[1]=10.1
            node._contact_wheel_samples.append((1.1,10.1,np.array([.2,.2,.3])))
        result=self.call(node,clock,receive)
        self.assertEqual(result[0],1.1)
        self.assertEqual(node._contact_wheel_samples[0][1],8.)
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_stale_sample_remains_rejected_after_bounded_wait(self):
        node,clock=self.fixture(stamp=.5)
        def tick(_):
            clock[0]+=.3; clock[1]+=.1
        with self.assertRaisesRegex(ValueError,'bounded stopped'):
            self.call(node,clock,tick)
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
        self.assertEqual(node._feedback_measurement_pause_generation,0)

    def test_paused_clock_is_bounded_by_wall_time(self):
        node,clock=self.fixture(stamp=1.05)
        def tick(_):
            clock[1]+=10.
        with self.assertRaisesRegex(ValueError,'bounded stopped'):
            self.call(node,clock,tick)
        self.assertEqual(node._contact_wheel_samples[0][0],1.05)

    def test_shutdown_interrupts_missing_pose_wait_without_motion(self):
        node,clock=self.fixture(stamp=.5)
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=False),\
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',return_value=10.):
            with self.assertRaisesRegex(ValueError,'interrupted'):
                node._feedback_wait_wheel_pose_sample()
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)

    def test_pulse_reobservation_preserves_pause_generation(self):
        node,clock=self.fixture()
        node._feedback_wait_wheel_pose_sample=Mock(return_value=(1.,np.zeros(3)))
        node._contact_wheel_samples.extend((t,10.,np.zeros(3)) for t in (.9,.95))
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.bounded_probe_pulse_scale',return_value=.2):
            self.assertEqual(node._feedback_probe_pulse_scale(0.,.3,.05),.2)
        node._feedback_wait_wheel_pose_sample.assert_called_once()

    def test_recovered_wheel_packet_cannot_continue_a_pulse_with_old_barriers(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(ManipulationNode._feedback_probe_contact_motion)))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and ast.unparse(node.func) in ('self._feedback_probe_pose_sample','self._feedback_probe_pulse_scale')]
        guards=[node for node in ast.walk(tree) if isinstance(node,ast.If)
                and ast.unparse(node.test).startswith("getattr(self, '_feedback_measurement_pause_generation'")]
        for call in calls:
            # The initial pose belongs to the next probe, not an active pulse.
            matching=[g for g in guards if 0<g.lineno-call.lineno<8]
            if not matching:
                self.assertEqual(ast.unparse(call.func),'self._feedback_probe_pose_sample')
                continue
            guard=min(matching,key=lambda g:g.lineno)
            later=Mock(side_effect=AssertionError('Old barriers cannot authorize motion after wait'))
            block=ast.Module(body=[ast.While(test=ast.Constant(True),body=[guard,
                ast.Expr(ast.Call(func=ast.Name('later',ast.Load()),args=[],keywords=[]))],orelse=[])],type_ignores=[])
            namespace=dict(self=Mock(_feedback_measurement_pause_generation=1),generation=0,
                           sensor_interrupted=False,later=later)
            exec(compile(ast.fix_missing_locations(block),'<actual_interruption_guard>','exec'),namespace)
            self.assertTrue(namespace['sensor_interrupted'])
            later.assert_not_called()

    def test_velocity_projection_discards_inputs_from_before_wheel_reobservation(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(RobotObservationManipulation._feedback_contact_velocity)))
        guard=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
            and ast.unparse(node.test).startswith("getattr(self, '_feedback_measurement_pause_generation'"))
        node=Mock(_feedback_measurement_pause_generation=1)
        block=ast.Module(body=[guard],type_ignores=[])
        with self.assertRaisesRegex(SceneObservationPending,'new contact velocity barriers'):
            exec(compile(ast.fix_missing_locations(block),'<actual_velocity_reobservation_guard>','exec'),
                 dict(self=node,generation=0,SceneObservationPending=SceneObservationPending))
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)


if __name__=='__main__':
    unittest.main()
