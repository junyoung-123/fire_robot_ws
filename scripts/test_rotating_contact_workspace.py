import math
import unittest
import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import (
    PiperActualKinematics, PIPER_JOINT_LIMITS, PIPER_STOW)
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation


class RotatingWorkspaceTests(unittest.TestCase):
    def test_compact_dock_selects_minimum_reach_across_camera_safe_lanes(self):
        arm=PiperActualKinematics()
        camera=np.array([.25,.0175,.789])
        lanes=[.245,*np.linspace(camera[1],arm.mount_xyz[1],17)]
        goal=arm.docking_contact_posture([1.2,.245,.764],.4628,
            camera_xyz=camera,camera_half_fov=.562,observed_span_m=.07,
            candidate_lanes=lanes,prefer_compact=True)
        self.assertLess(goal[0],.53)
        self.assertLess(goal[1],.15)
        self.assertGreaterEqual(goal[0],.4628)
        self.assertLessEqual(abs(goal[1]-camera[1])+.035,(goal[0]-camera[0])*np.tan(.562)+1.e-12)
        solution=arm.solve(goal,max_iterations=100,position_tolerance_m=.003,
            minimum_joint_margin_rad=.20)
        self.assertIsNotNone(solution)
        self.assertGreaterEqual(arm.translation_condition_ratio(solution.positions),.25)

    def test_contact_docking_prefers_mount_lane_without_clipping_observed_handle(self):
        arm=PiperActualKinematics()
        camera=np.array([.25,.0175,.789])
        span=.3615006692433463
        for height in (.7657,.8):
            goal=arm.docking_contact_posture([1.5,.15,height],.4628,
                camera_xyz=camera,camera_half_fov=.562,observed_span_m=span,prefer_mount_lane=True)
            self.assertGreater(goal[1],.10)
            self.assertLessEqual(abs(goal[1]-camera[1])+span/2.,(goal[0]-camera[0])*np.tan(.562)+1.e-12)
            solution=arm.solve(goal,max_iterations=100,position_tolerance_m=.003,
                               minimum_joint_margin_rad=.20)
            self.assertIsNotNone(solution)
            self.assertGreaterEqual(arm.translation_condition_ratio(solution.positions),.25)

    def test_joint_unfolding_avoids_cartesian_ik_hole(self):
        arm=PiperActualKinematics()
        goal=np.array([.53737,.04842,.76426])
        path=arm.joint_approach_path(goal,PIPER_STOW,.61006)
        self.assertTrue(np.all(path >= PIPER_JOINT_LIMITS[:,0]))
        self.assertTrue(np.all(path <= PIPER_JOINT_LIMITS[:,1]))
        self.assertLessEqual(np.max(np.abs(np.diff(path,axis=0))),.025)
        self.assertLess(np.linalg.norm(arm.forward(path[-1])[0]-goal),.003)
        noisy=PIPER_STOW.copy()
        noisy[1]=-.00005
        self.assertTrue(np.all(arm.joint_approach_path(goal,noisy,.61006) >= PIPER_JOINT_LIMITS[:,0]))
        with self.assertRaisesRegex(ValueError,'panel plane'):
            arm.joint_approach_path(goal,PIPER_STOW,.50)

    def test_dock_before_far_edge_of_reach(self):
        arm = PiperActualKinematics()
        minimum = .312 + arm.tool_front_extent_m + .015
        for height in (.73,.76,.80):
            observed = np.array([.695,.145,height])
            preferred = arm.docking_contact_posture(observed, minimum)
            self.assertLess(preferred[0], observed[0]-.10)
            np.testing.assert_array_equal(preferred[1:], observed[1:])
            self.assertGreaterEqual(preferred[0], minimum)

    def test_camera_and_arm_share_visible_close_docking_pose(self):
        arm=PiperActualKinematics()
        camera=np.array([.25,.0175,.789])
        for height in (.73,.76,.80):
            preferred=arm.docking_contact_posture(
                [.695,.145,height],.4628,camera_xyz=camera,
                camera_half_fov=.562,observed_span_m=.25)
            self.assertLess(preferred[0],.655)
            self.assertLessEqual(abs(preferred[1]-camera[1])+.125,
                                 (preferred[0]-camera[0])*np.tan(.562))

    def test_arm_workspace_is_not_limited_to_chassis_width(self):
        arm = PiperActualKinematics()
        seed = np.array([-.3,2.,-1.5,.8,-.6,.6])
        rotation = measured_wrist_rotation(arm.approach_rotation,-1.1,0.,-.32)
        future = measured_wrist_rotation(rotation,-.35,0.)
        preferred, _ = arm.preferred_contact_posture(
            [.695,.145,.73],rotation,seed,.463,future_rotation=future)
        self.assertLess(preferred[1],-.226)

    def test_posture_reserves_joint_margin_for_turning_panel(self):
        arm = PiperActualKinematics()
        seed = np.array([0.,1.8,-1.2,0.,-.65,1.5])
        observed = np.array([.695,.145,.73])
        for angle in (0., -.2, -.35, -.45):
            rotation = measured_wrist_rotation(arm.approach_rotation, angle, 0., -.32)
            preferred, _ = arm.preferred_contact_posture(observed, rotation, seed, .503)
            solution = arm.solve(preferred, seed=seed, target_rotation=rotation,
                                 position_tolerance_m=.003, prefer_seed_solution=True)
            self.assertIsNotNone(solution)
            margin = np.min(np.minimum(solution.positions-PIPER_JOINT_LIMITS[:,0],
                                       PIPER_JOINT_LIMITS[:,1]-solution.positions))
            self.assertGreaterEqual(margin, .20)
            seed = solution.positions


if __name__ == '__main__':
    unittest.main()
