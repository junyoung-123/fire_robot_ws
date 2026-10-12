import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from fire_robot_manipulation.contact_feedback import observed_panel_docking_lanes
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_BODY_OUTLINE


class ObservedPanelDockingTests(unittest.TestCase):
    def test_partial_panel_view_turns_toward_observed_midpoint(self):
        from fire_robot_manipulation.mission_contact_adapter import panel_view_command
        self.assertEqual(panel_view_command([1., -.4], [.25, .02], 0.), (0., -.12))
        self.assertEqual(panel_view_command([1., .4], [.25, .02], 0.), (0., .12))

    def test_centred_partial_view_backs_away_to_reobserve_edges(self):
        from fire_robot_manipulation.mission_contact_adapter import panel_view_command
        self.assertEqual(panel_view_command([1., .02], [.25, .02], 0.), (-.04, 0.))

    def test_view_command_respects_calibrated_camera_direction(self):
        from fire_robot_manipulation.mission_contact_adapter import panel_view_command
        self.assertEqual(panel_view_command([0., 1.], [0., .25], math.pi/2), (-.04, 0.))

    def test_invalid_or_too_close_panel_does_not_authorize_view_motion(self):
        from fire_robot_manipulation.mission_contact_adapter import panel_view_command
        for point in ([float('nan'), 1.], [.3, 0.]):
            with self.assertRaises(ValueError):
                panel_view_command(point, [.25, 0.], 0.)

    def test_view_motion_requires_fresh_whole_body_sweep(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        from fire_robot_manipulation.mission_contact_adapter import MissionContactAdapter
        for scene in (None, NS(safe=lambda v,w:False), NS(safe=lambda v,w:True)):
            node = NS(view_guard=NS(snapshot=lambda *args:scene),
                      scan=None,scan_received=10.,guarded_cmd=Mock())
            applied = MissionContactAdapter.guarded_view_command(node, 0., -.12)
            self.assertEqual(applied, (0., -.12) if scene is not None and scene.safe(0.,-.12) else (0., 0.))

    def current_point_fixture(self, exposure=10., transform_stamp=10.15):
        from types import SimpleNamespace
        from geometry_msgs.msg import TransformStamped
        from fire_robot_interfaces.msg import DoorInfo
        from fire_robot_manipulation.mission_contact_adapter import MissionContactAdapter
        observed=DoorInfo()
        observed.handle_position.header.frame_id='base_link'
        observed.handle_position.header.stamp.sec=10
        observed.handle_position.point.x=.54
        observed.handle_position.point.y=.14
        observed.handle_position.point.z=.77
        transform=TransformStamped()
        transform.header.frame_id='base_link'
        transform.header.stamp.sec=int(transform_stamp)
        transform.header.stamp.nanosec=round((transform_stamp-int(transform_stamp))*1.e9)
        transform.child_frame_id='odom'
        transform.transform.translation.x=-.02
        transform.transform.rotation.w=1.
        instance=SimpleNamespace(now=lambda:10.2,reference_frame='odom',
            tf=SimpleNamespace(lookup_transform=lambda *args:transform))
        latest=(observed,exposure,np.array([.54,.14,.77]))
        return MissionContactAdapter.current_handle_xyz,instance,latest,transform

    def test_current_alignment_coordinates_compensate_motion_since_exposure(self):
        method,instance,latest,_=self.current_point_fixture()
        current=method(instance,latest)
        np.testing.assert_allclose(current,[.52,.14,.77])
        self.assertEqual(latest[0].handle_position.point.x,.54)
        self.assertEqual(latest[0].handle_position.header.stamp.sec,10)
        self.assertEqual(latest[0].handle_position.header.stamp.nanosec,0)

    def test_current_frame_never_refreshes_stale_or_future_exposure(self):
        for stamp in (9.3,10.3):
            method,instance,latest,_=self.current_point_fixture(exposure=stamp)
            with self.assertRaisesRegex(ValueError,'exposure'):
                method(instance,latest)

    def test_current_frame_requires_fresh_robot_transform(self):
        for stamp in (10.,10.3):
            method,instance,latest,_=self.current_point_fixture(transform_stamp=stamp)
            with self.assertRaisesRegex(ValueError,'transform'):
                method(instance,latest)

    def test_rotated_current_robot_frame_is_not_the_exposure_frame(self):
        method,instance,latest,transform=self.current_point_fixture()
        transform.transform.rotation.w=math.sqrt(.5)
        transform.transform.rotation.z=-math.sqrt(.5)
        transform.transform.translation.x=0.
        np.testing.assert_allclose(method(instance,latest),[.14,-.54,.77],atol=1.e-12)

    def test_panel_centre_not_arm_mount_determines_the_first_lane(self):
        lanes=observed_panel_docking_lanes([0.,.66],[1.,0.],[[0.,0.],[0.,.8]],
            PIPER_BODY_OUTLINE,.05)
        self.assertAlmostEqual(lanes[0],.26)
        for lane in lanes:
            body_low=.66-lane+PIPER_BODY_OUTLINE[:,1].min()
            body_high=.66-lane+PIPER_BODY_OUTLINE[:,1].max()
            self.assertGreaterEqual(body_low,.05-1.e-9)
            self.assertLessEqual(body_high,.75+1.e-9)

    def test_world_origin_and_orientation_do_not_change_lanes(self):
        original=observed_panel_docking_lanes([0.,.66],[1.,0.],[[0.,0.],[0.,.8]],
            PIPER_BODY_OUTLINE,.05)
        a=1.31
        rotate=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        shift=np.array([31.,-15.])
        lanes=observed_panel_docking_lanes(rotate@[0.,.66]+shift,rotate@[1.,0.],
            np.array([[0.,0.],[0.,.8]])@rotate.T+shift,PIPER_BODY_OUTLINE,.05)
        self.assertAlmostEqual(lanes[0],original[0])
        np.testing.assert_allclose(sorted(set(lanes)),sorted(set(original)),atol=1.e-12)

    def test_unobserved_narrow_or_unrelated_support_is_not_an_aperture(self):
        for segment in ([[0.,0.],[0.,.3]],[[0.,-2.],[0.,-1.]]):
            with self.assertRaises(ValueError):
                observed_panel_docking_lanes([0.,.66],[1.,0.],segment,PIPER_BODY_OUTLINE,.05)

    def test_centred_grasp_is_in_real_piper_reach_and_camera_fov(self):
        arm=PiperActualKinematics()
        camera=np.array([.25,.0175,.789])
        lanes=observed_panel_docking_lanes([0.,.66],[1.,0.],[[0.,0.],[0.,.8]],
            PIPER_BODY_OUTLINE,.05)
        goal=arm.docking_contact_posture([1.5,.15,.768],.4628,camera_xyz=camera,
            camera_half_fov=.562,observed_span_m=.07,candidate_lanes=lanes)
        self.assertAlmostEqual(goal[1],lanes[0])
        self.assertLessEqual(abs(goal[1]-camera[1])+.035,(goal[0]-camera[0])*math.tan(.562))
        solution=arm.solve(goal,max_iterations=100,position_tolerance_m=.003,
                           minimum_joint_margin_rad=.20)
        self.assertIsNotNone(solution)
        self.assertGreaterEqual(arm.translation_condition_ratio(solution.positions),.25)

    def test_visibility_width_matches_actual_gripper_joint_limits(self):
        path=Path(__file__).resolve().parents[1]/'src/fire_robot_description/urdf/piper_arm_actual.xacro'
        root=ET.parse(path).getroot()
        widths=[float(j.find('limit').get('upper'))-float(j.find('limit').get('lower'))
            for j in root.findall('.//joint') if j.get('name') in ('gripper_left_joint','gripper_right_joint')]
        self.assertAlmostEqual(sum(widths),.07)

    def test_recorded_partial_panel_support_does_not_select_camera_centred_edge(self):
        arm = PiperActualKinematics()
        lanes = observed_panel_docking_lanes([.93, .051], [1., 0.],
            [[.93, -.429], [.93, .181]], PIPER_BODY_OUTLINE, .05)
        goal = arm.docking_contact_posture([1.28, .15, .765], .4628,
            camera_xyz=[.25, .0175, .789], camera_half_fov=.562,
            observed_span_m=.07, candidate_lanes=lanes, prefer_compact=True)
        self.assertGreater(goal[1], .050625)
        self.assertGreaterEqual(.051-goal[1]+PIPER_BODY_OUTLINE[:,1].min(), -.379-1.e-9)
        self.assertLessEqual(.051-goal[1]+PIPER_BODY_OUTLINE[:,1].max(), .131+1.e-9)

    def test_wide_discovery_roi_does_not_force_an_extended_near_grasp(self):
        arm=PiperActualKinematics()
        lanes=[.12484475204603143,.15478831919593067,.18473188634582988]
        options=dict(camera_xyz=[.25,.0175,.789],camera_half_fov=.562,
                     candidate_lanes=lanes,prefer_compact=True)
        full=arm.docking_contact_posture([1.3,.15,.763],.4628,
            observed_span_m=.3402247138082962,**options)
        grasp=arm.docking_contact_posture([1.3,.15,.763],.4628,
            observed_span_m=min(.3402247138082962,.07),**options)
        self.assertLess(grasp[0],full[0]-.1)
        self.assertIn(grasp[1],lanes)
        self.assertLessEqual(abs(grasp[1]-.0175)+.035,
                            (grasp[0]-.25)*math.tan(.562))
        solution=arm.solve(grasp,max_iterations=100,position_tolerance_m=.003,
                           minimum_joint_margin_rad=.20)
        self.assertIsNotNone(solution)
        self.assertGreaterEqual(arm.translation_condition_ratio(solution.positions),.25)

    def test_near_grasp_view_keeps_live_primary_detection_and_depth_guards(self):
        import inspect
        from fire_robot_manipulation import mission_contact_adapter as adapter
        source=inspect.getsource(adapter.MissionContactAdapter.park_from_observation)
        self.assertIn('visibility_span=min(span,.07)',source)
        self.assertIn('primary_confidence_gate=.45',source)
        self.assertIn('candidate_lanes=lanes',source)
        self.assertIn("self.now()-self.panel['depth_stamp'] > .6",source)

    def test_camera_reacquisition_does_not_bypass_near_docking_distance(self):
        import ast
        import inspect
        import textwrap
        from fire_robot_manipulation import mission_contact_adapter as adapter
        source=inspect.getsource(adapter.MissionContactAdapter.open)
        self.assertNotIn('or camera_reacquired',source)
        tree=ast.parse(textwrap.dedent(source))
        assignments=[node.value for node in ast.walk(tree)
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name)
            and target.id=='reachable' for target in node.targets)]
        self.assertTrue(any('docking_posture' in ast.unparse(value)
                            and 'xyz[0]' in ast.unparse(value) for value in assignments))

    def test_adapter_never_merges_camera_only_lanes_into_panel_supported_lanes(self):
        import ast
        from fire_robot_manipulation import mission_contact_adapter as adapter
        import inspect
        import textwrap
        source = inspect.getsource(adapter.MissionContactAdapter.park_from_observation)
        tree = ast.parse(textwrap.dedent(source))
        self.assertNotIn('calibrated_lanes', source)
        self.assertIn("candidate_lanes=lanes", source)
        self.assertIn('panel_docking_support_unavailable', source)
        self.assertTrue(any(isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
                            and node.value.value is False for node in ast.walk(tree)))


if __name__=='__main__':
    unittest.main()
