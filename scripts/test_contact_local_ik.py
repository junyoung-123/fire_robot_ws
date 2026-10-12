import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np

from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics,PIPER_JOINT_LIMITS
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class ContactLocalIkTests(unittest.TestCase):
    def test_reachable_small_update_keeps_pose_tolerances_and_seed_region(self):
        arm=PiperActualKinematics()
        seed=np.array([.3,1.2,-1.5,.1,.2,-.1])
        target,rotation=arm.forward(seed+[.02,.01,-.01,.02,.01,-.015])
        solution=arm.solve(target,seed=seed,target_rotation=rotation,max_iterations=80,
            position_tolerance_m=.001,orientation_tolerance_rad=.03,maximum_seed_step_rad=.14)
        self.assertIsNotNone(solution)
        self.assertLessEqual(solution.position_error_m,.001)
        self.assertLessEqual(solution.orientation_error_rad,.03)
        self.assertTrue(np.all(solution.positions>=seed-.14))
        self.assertTrue(np.all(solution.positions<=seed+.14))

    def test_far_target_cannot_use_a_different_global_configuration(self):
        arm=PiperActualKinematics()
        seed=np.array([.3,1.2,-1.5,.1,.2,-.1])
        target,rotation=arm.forward(seed+[.8,0.,0.,0.,0.,0.])
        result=arm.solve(target,seed=seed,target_rotation=rotation,max_iterations=80,
            position_tolerance_m=.001,orientation_tolerance_rad=.03,maximum_seed_step_rad=.14)
        self.assertIsNone(result)

    def test_invalid_seed_or_enlarged_step_is_rejected(self):
        arm=PiperActualKinematics()
        for seed,step in ((None,.14),(np.full(6,np.nan),.14),
                          (PIPER_JOINT_LIMITS[:,1]+.01,.14),(np.zeros(6),.141),
                          (np.zeros(6),0.),(np.zeros(6),math.nan)):
            with self.subTest(seed=seed,step=step):
                self.assertIsNone(arm.solve([.5,0.,.7],seed=seed,maximum_seed_step_rad=step))

    def test_seed_region_does_not_override_real_urdf_limits(self):
        arm=PiperActualKinematics()
        seed=PIPER_JOINT_LIMITS[:,1]-.01
        target,rotation=arm.forward(seed)
        result=arm.solve(target,seed=seed,target_rotation=rotation,maximum_seed_step_rad=.14)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result.positions<=PIPER_JOINT_LIMITS[:,1]))
        self.assertTrue(np.all(result.positions>=PIPER_JOINT_LIMITS[:,0]))

    def fixture(self):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        clock=[1.,10.]
        node._sim_time_sec=lambda:clock[0]
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        node._feedback_measurement_pause_generation=0
        return node,clock

    def observe(self,node,clock):
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',side_effect=lambda:clock[1]):
            return node._feedback_contact_ik_reobservation([.5,0.,.7],np.eye(3),np.zeros(6),None)

    def test_failed_local_ik_stops_and_requires_new_barriers(self):
        node,clock=self.fixture()
        self.assertTrue(self.observe(node,clock))
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)
        self.assertEqual(node._feedback_measurement_pause_generation,1)
        self.assertEqual(node._feedback_event.call_args.kwargs['maximum_joint_step_rad'],.14)

    def test_repeated_failed_ik_is_bounded_in_sim_and_wall_time(self):
        for sim,wall in ((3.01,10.1),(1.,40.01)):
            node,clock=self.fixture()
            self.assertTrue(self.observe(node,clock))
            clock[:]=[sim,wall]
            self.assertFalse(self.observe(node,clock))
            self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
            self.assertEqual(node._feedback_measurement_pause_generation,1)


if __name__=='__main__':
    unittest.main()
