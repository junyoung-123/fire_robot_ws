from collections import deque
from types import SimpleNamespace, MethodType
import unittest
import json
import math
from unittest.mock import patch, Mock
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tf2_ros import TransformException
from fire_robot_manipulation.contact_scan_localization_node import ContactScanLocalization
from fire_robot_manipulation.contact_scan_registration import ScanPose, CorrectionFilter
from fire_robot_manipulation.mission_contact_adapter import MissionContactAdapter
from fire_robot_manipulation.observed_door_angle_node import ObservedDoorAngle
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from rclpy.time import Time


class ContactLocalizationLifecycleTests(unittest.TestCase):
    def propagation_stub(self, valid=True, measurement=1., received=10.):
        published = []
        node = SimpleNamespace(scan_fit_valid=valid, trusted_scan_stamp=measurement,
            trusted_scan_received=received, publish=lambda stamp: published.append(stamp),
            report=lambda *args, **kwargs: None,
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1100000000)))
        node.process_wheel = MethodType(ContactScanLocalization.process_wheel, node)
        message = SimpleNamespace(header=SimpleNamespace(frame_id='odom',
            stamp=SimpleNamespace(sec=1, nanosec=100000000)), child_frame_id='base_link')
        return node, message, published

    def test_fresh_scan_correction_uses_actual_wheel_stamp_without_restamping_scan(self):
        node, msg, published = self.propagation_stub()
        with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic', return_value=10.1):
            ContactScanLocalization.wheel_cb(node, msg)
        self.assertEqual(published, [msg.header.stamp])
        self.assertEqual(node.trusted_scan_stamp, 1.)
        self.assertEqual(node.trusted_scan_received, 10.)

    def test_actual_robot_urdf_wheel_message_frame_is_accepted(self):
        path = Path(__file__).resolve().parents[1] / 'src/fire_robot_description/urdf/fire_robot_actual_piper.urdf.xacro'
        robot = ET.parse(path).getroot()
        drive = next(plugin for plugin in robot.findall('.//plugin')
                     if plugin.attrib.get('name', '').endswith('::DiffDrive'))
        node, msg, published = self.propagation_stub()
        msg.header.frame_id = drive.findtext('frame_id')
        msg.child_frame_id = drive.findtext('child_frame_id')
        self.assertEqual(msg.child_frame_id, 'base_footprint')
        with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic', return_value=10.1):
            ContactScanLocalization.wheel_cb(node, msg)
        self.assertEqual(published, [msg.header.stamp])

    def test_propagation_audit_keeps_original_scan_measurement_time(self):
        node, msg, _ = self.propagation_stub()
        records = []
        node.report = lambda stamp, valid, **values: records.append((stamp, valid, values))
        with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic', return_value=10.1):
            ContactScanLocalization.wheel_cb(node, msg)
        self.assertEqual(records[0][0], 1.)
        self.assertAlmostEqual(records[0][2]['pose_stamp'], 1.1)
        self.assertAlmostEqual(records[0][2]['correction_measurement_age_sec'], .1)

    def test_failed_or_stale_scan_never_propagates_wheel_feedback(self):
        for kwargs in (dict(valid=False), dict(measurement=.8), dict(received=8.)):
            node, msg, published = self.propagation_stub(**kwargs)
            with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic', return_value=10.1):
                ContactScanLocalization.wheel_cb(node, msg)
            self.assertEqual(published, [])

    def test_future_wheel_sample_and_wrong_frame_never_propagate(self):
        node, msg, published = self.propagation_stub()
        msg.header.stamp.nanosec = 200000000
        with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic', return_value=10.1):
            ContactScanLocalization.wheel_cb(node, msg)
            msg.header.stamp.nanosec = 100000000
            msg.header.frame_id = 'evaluation'
            ContactScanLocalization.wheel_cb(node, msg)
        self.assertEqual(published, [])

    def test_clock_lag_waits_without_changing_measurement_timestamp(self):
        node, msg, published = self.propagation_stub()
        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1090000000))
        with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic', return_value=10.1):
            ContactScanLocalization.wheel_cb(node, msg)
            self.assertEqual(published, [])
            node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1100000000))
            node.process_wheel()
        self.assertEqual(published, [msg.header.stamp])
        self.assertEqual(node.trusted_scan_stamp, 1.)

    def test_reference_is_not_kept_during_next_navigation_job(self):
        stub = SimpleNamespace(correction=np.array([.02, -.01, .01]), reference=object(),
                               freeze_requested=True)
        ContactScanLocalization.phase_cb(stub, SimpleNamespace(data='COMPLETE:observed_id'))
        self.assertIsNone(stub.reference)
        self.assertFalse(stub.freeze_requested)
        np.testing.assert_equal(stub.correction, [.02, -.01, .01])

    def test_freeze_starts_before_arm_contact(self):
        stub = SimpleNamespace(freeze_requested=False)
        ContactScanLocalization.phase_cb(stub, SimpleNamespace(data='PRE_GRASP:observed_id'))
        self.assertTrue(stub.freeze_requested)

    def test_closed_reference_collection_starts_before_initial_angle_gate(self):
        stub=SimpleNamespace(correction=np.zeros(3),reference=object(),freeze_requested=False)
        ContactScanLocalization.phase_cb(stub,SimpleNamespace(data='LOCALIZE_HANDLE:observed_id'))
        self.assertIsNone(stub.reference)
        self.assertTrue(stub.freeze_requested)
        self.assertEqual(stub.phase,'LOCALIZE_HANDLE')

    def test_contact_cannot_use_wheel_only_pose_without_frozen_observed_reference(self):
        stamp=Time(seconds=10.).to_msg()
        msg=SimpleNamespace(header=SimpleNamespace(stamp=stamp,frame_id='lidar_link'),
            ranges=[1.]*100,angle_min=0.,angle_increment=.01,range_min=.15,range_max=12.)
        transform=SimpleNamespace(translation=SimpleNamespace(x=0.,y=0.,z=0.),
            rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.))
        node=SimpleNamespace(pending=msg,last_stamp=9.,correction=np.zeros(3),reference=None,
            static_scans=deque(),freeze_requested=True,phase='PRESS_LEVER',
            process_wheel=lambda:None,exclusion_panel=lambda _:None,
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10000000000)),
            tf=Mock(),report=Mock(),trust_scan=Mock(),publish=Mock())
        node.tf.lookup_transform.return_value=SimpleNamespace(transform=transform)
        ContactScanLocalization.process(node)
        node.trust_scan.assert_not_called()
        node.publish.assert_not_called()
        self.assertFalse(node.scan_fit_valid)
        self.assertEqual(node.report.call_args.kwargs['reason'],'static_reference_not_ready_before_contact')

    def scan_fixture(self,translation_uncertainty=.04,translation_delta=.04,yaw=.005):
        stamp=Time(seconds=10.).to_msg()
        msg=SimpleNamespace(header=SimpleNamespace(stamp=stamp,frame_id='lidar_link'),
            ranges=[1.]*100,angle_min=0.,angle_increment=.01,range_min=.15,range_max=12.)
        transform=SimpleNamespace(translation=SimpleNamespace(x=0.,y=0.,z=0.),
            rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.))
        reference=Mock()
        reference.estimate.return_value=ScanPose(np.array([translation_delta,0.,yaw]),
            .02,translation_uncertainty,.002,100)
        node=SimpleNamespace(pending=msg,last_stamp=9.,correction=np.zeros(3),reference=reference,
            filter=CorrectionFilter(np.zeros(3)),process_wheel=lambda:None,
            exclusion_panel=lambda _:dict(valid=True,depth_stamp=10.,center=[2.,0.,.8],
                normal=[1.,0.,0.],residual_m=.005,orientation_uncertainty_rad=.01),
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10000000000)),
            tf=Mock(),report=Mock(),trust_scan=Mock(),publish=Mock())
        node.tf.lookup_transform.return_value=SimpleNamespace(transform=transform)
        return node,msg

    def test_uncertified_scan_translation_is_not_applied_or_reported_as_full_pose(self):
        node,msg=self.scan_fixture()
        ContactScanLocalization.process(node)
        np.testing.assert_allclose(node.correction,[0.,0.,.005])
        node.publish.assert_called_once_with(msg.header.stamp)
        node.reference.estimate.assert_called_once()
        self.assertFalse(node.report.call_args.kwargs['full_pose_valid'])
        self.assertIsNone(node.report.call_args.kwargs['translation_uncertainty_m'])
        self.assertEqual(node.report.call_args.kwargs['unapplied_scan_translation_uncertainty_m'],.04)

    def test_large_scan_translation_jump_cannot_change_encoder_translation(self):
        node,_=self.scan_fixture(translation_uncertainty=.01,translation_delta=.12)
        ContactScanLocalization.process(node)
        np.testing.assert_allclose(node.correction,[0.,0.,.005])
        self.assertFalse(node.report.call_args.kwargs['full_pose_valid'])
        self.assertEqual(node.report.call_args.kwargs['unapplied_scan_translation_delta_m'],.12)

    def test_full_qualified_scan_retains_original_translation_correction(self):
        node,_=self.scan_fixture(translation_uncertainty=.01,translation_delta=.02)
        ContactScanLocalization.process(node)
        np.testing.assert_allclose(node.correction,[.02,0.,.005])
        self.assertTrue(node.report.call_args.kwargs['full_pose_valid'])

    def test_actual_ros_status_json_serializes_both_pose_quality_modes(self):
        for uncertainty in (.01,.04):
            node,_=self.scan_fixture(translation_uncertainty=uncertainty,translation_delta=.02)
            node.status=Mock()
            node.report=MethodType(ContactScanLocalization.report,node)
            ContactScanLocalization.process(node)
            payload=json.loads(node.status.publish.call_args.args[0].data)
            self.assertIsInstance(payload['full_pose_valid'],bool)
            self.assertEqual(payload['full_pose_valid'],uncertainty==.01)
            self.assertTrue(payload['valid'])
            self.assertFalse(payload['environment_truth_used'])

    def test_bad_or_lagged_yaw_does_not_commit_filter_or_publish_tf(self):
        for yaw in (.05,.034):
            node,_=self.scan_fixture(yaw=yaw)
            if yaw==.034:
                node.filter.update(np.zeros(3),9.9)
            old_filter=node.filter
            ContactScanLocalization.process(node)
            np.testing.assert_equal(node.correction,np.zeros(3))
            self.assertIs(node.filter,old_filter)
            node.publish.assert_not_called()
            node.trust_scan.assert_not_called()
            self.assertFalse(node.scan_fit_valid)

    def test_propagated_yaw_quality_does_not_promote_translation_to_certified_pose(self):
        node,msg,_=self.propagation_stub()
        node.trusted_scan_quality=dict(full_pose_valid=False,translation_uncertainty_m=None,
            yaw_uncertainty_rad=.01,translation_source='wheel_encoders_with_last_qualified_translation_correction')
        node.report=Mock()
        with patch('fire_robot_manipulation.contact_scan_localization_node.time.monotonic',return_value=10.1):
            ContactScanLocalization.wheel_cb(node,msg)
        self.assertFalse(node.report.call_args.kwargs['full_pose_valid'])
        self.assertIsNone(node.report.call_args.kwargs['translation_uncertainty_m'])

    def test_door_angle_confidence_includes_current_and_closed_reference_yaw(self):
        node=SimpleNamespace(reference_frame='contact_scan',baseline_yaw_uncertainty=None,
            reference_yaw_history=deque([(10.,.004)]))
        self.assertAlmostEqual(ObservedDoorAngle.angle_uncertainty(node,10.1,.005),math.hypot(.005,.008))
        node.reference_yaw_history.append((11.,.006))
        self.assertAlmostEqual(ObservedDoorAngle.angle_uncertainty(node,11.1,.005),math.hypot(.005,.010))

    def test_stale_future_or_overbound_reference_yaw_cannot_certify_panel_angle(self):
        for stamp,current in ((9.,.004),(10.2,.004),(10.,.034)):
            node=SimpleNamespace(reference_frame='contact_scan',baseline_yaw_uncertainty=.004,
                reference_yaw_history=deque([(stamp,current)]))
            with self.assertRaises(ValueError):
                ObservedDoorAngle.angle_uncertainty(node,10.1,.005)

    def test_closed_plane_baseline_resets_only_at_measured_precontact_reference(self):
        node=SimpleNamespace(tracker=object(),baseline_yaw_uncertainty=.01,
            reference_yaw_history=deque([(8.,.01)]))
        value=dict(valid=True,event='static_reference_frozen',measurement_stamp=10.,
            yaw_uncertainty_rad=math.radians(.2),full_pose_valid=False)
        ObservedDoorAngle.localization_cb(node,SimpleNamespace(data=json.dumps(value)))
        self.assertIsNone(node.tracker.reference)
        self.assertIsNone(node.baseline_yaw_uncertainty)
        self.assertEqual(list(node.reference_yaw_history),[(10.,math.radians(.2))])

    def test_initial_geometry_can_initialize_scan_reference_but_not_opening_certificate(self):
        observer=SimpleNamespace(report=Mock())
        fit=dict(normal=np.array([1.,0.,0.]),center=np.array([.5,.2,.8]),
            segment_xy=np.array([[.5,-.2],[.5,.6]]),residual_m=.005,
            orientation_uncertainty_rad=.01,inliers=100)
        ObservedDoorAngle.publish_initial_geometry(observer,fit,10.,'recent_yolo_handle')
        payload=dict(valid=True,depth_stamp=10.,reference_frame='contact_scan',
            **observer.report.call_args.kwargs)
        self.assertNotIn('angle_rad',payload)
        msg=SimpleNamespace(data=json.dumps(payload))
        localizer=SimpleNamespace(panel=None)
        ContactScanLocalization.panel_cb(localizer,msg)
        self.assertIsNotNone(localizer.panel)
        controller=SimpleNamespace(_contact_reference_frame='contact_scan',
            _plane_confirmations=deque(),_sim_door_state_callback=Mock())
        RobotObservationManipulation._panel_status_callback(controller,msg)
        self.assertEqual(len(controller._plane_confirmations),0)
        controller._sim_door_state_callback.assert_not_called()
        self.assertFalse(hasattr(controller,'_depth_confirmation'))

    def test_camera_stamp_is_preserved_while_waiting_for_scan_tf(self):
        point, transformed = object(), object()
        calls = []
        def transform(value, frame):
            calls.append((value, frame))
            if len(calls) == 1:
                raise TransformException('TF is 3ms behind camera')
            return transformed
        stopped = []
        stub = SimpleNamespace(now=lambda: 1., transform=transform,
                               reference_frame='contact_scan', command=lambda: stopped.append(True))
        with patch('fire_robot_manipulation.mission_contact_adapter.rclpy.ok', return_value=True):
            result = MissionContactAdapter.wait_transform(stub, point)
        self.assertIs(result, transformed)
        self.assertEqual(calls, [(point, 'contact_scan'), (point, 'contact_scan')])
        self.assertEqual(len(stopped), 2)

    def local_exclusion_fixture(self):
        panel=dict(valid=True,depth_stamp=10.,reference_frame='base_link',center=[.5,.2,.8],
            normal=[1.,0.,0.],segment_xy=[[.5,-.2],[.5,.6]],
            source='registered_rgbd_and_own_wrist_fk_exclusion_only')
        q=SimpleNamespace(x=0.,y=0.,z=np.sin(np.pi/4),w=np.cos(np.pi/4))
        tf=SimpleNamespace(transform=SimpleNamespace(rotation=q,
            translation=SimpleNamespace(x=.3,y=-.1,z=0.)))
        node=SimpleNamespace(local_panel=panel,panel={'depth_stamp':8.},correction=np.array([.1,0.,0.]),
            tf=Mock())
        node.tf.lookup_transform.return_value=tf
        return node,panel

    def test_local_panel_exclusion_uses_only_timestamped_own_wheel_tf(self):
        node,panel=self.local_exclusion_fixture()
        result=ContactScanLocalization.exclusion_panel(node,10.1)
        args=node.tf.lookup_transform.call_args.args
        self.assertEqual(args[:2],('odom','base_link'))
        self.assertEqual(args[2].nanoseconds,Time(seconds=10.).nanoseconds)
        np.testing.assert_allclose(result['center'],[.2,.4,.8],atol=1.e-12)
        np.testing.assert_allclose(result['normal'],[0.,1.,0.],atol=1.e-12)
        self.assertEqual(result['depth_stamp'],10.)
        self.assertEqual(panel['reference_frame'],'base_link')
        self.assertNotIn('angle_rad',result)

    def test_stale_or_future_local_panel_never_refreshes_exclusion(self):
        for stamp in (10.3,9.99):
            node,_=self.local_exclusion_fixture()
            self.assertIs(ContactScanLocalization.exclusion_panel(node,stamp),node.panel)
            node.tf.lookup_transform.assert_not_called()

    def test_missing_own_wheel_tf_never_uses_latest_global_pose(self):
        node,_=self.local_exclusion_fixture()
        node.tf.lookup_transform.side_effect=TransformException('wheel time pending')
        self.assertIs(ContactScanLocalization.exclusion_panel(node,10.1),node.panel)
        node.tf.lookup_transform.assert_called_once()

    def test_local_rgbd_fit_does_not_request_contact_scan_tf_or_publish_angle(self):
        node=ObservedDoorAngle.__new__(ObservedDoorAngle)
        header=SimpleNamespace(stamp=Time(seconds=10.1).to_msg(),frame_id='camera_color_optical_frame')
        msg=SimpleNamespace(header=header,encoding='32FC1')
        rgb=SimpleNamespace(header=header)
        node.info=SimpleNamespace(height=8,width=8,k=[100.,0.,4.,0.,100.,4.,0.,0.,1.])
        node.local_contact_stamp=10.
        node.local_contact_anchor=np.array([.5,.2,.8])
        node.local_contact_history=deque([(10.,node.local_contact_anchor.copy()),
            (10.2,np.array([.6,.2,.8]))],maxlen=40)
        node.rgbs=[rgb];node.panel_color='blue';node.rng=np.random.default_rng(1)
        node.bridge=Mock()
        node.bridge.imgmsg_to_cv2.side_effect=[np.ones((8,8),np.float32),np.zeros((8,8,3),np.uint8)]
        node.tf=Mock()
        node.tf.lookup_transform.return_value=SimpleNamespace(transform=SimpleNamespace(
            rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.),
            translation=SimpleNamespace(x=0.,y=0.,z=0.)))
        node.local_panel_pub=Mock();node.pub=Mock()
        fit=dict(center=np.array([.5,.2,.8]),normal=np.array([1.,0.,0.]),
            segment_xy=np.array([[.5,-.2],[.5,.6]]),residual_m=.005,
            orientation_uncertainty_rad=.01,inliers=100)
        with patch('fire_robot_manipulation.observed_door_angle_node.panel_color_support',
                   return_value=np.ones((8,8),bool)), \
             patch('fire_robot_manipulation.observed_door_angle_node.fit_panel',return_value=fit) as fit_mock:
            node.publish_local_panel(msg)
        np.testing.assert_equal(fit_mock.call_args.args[1],[.5,.2,.8])
        self.assertEqual(node.tf.lookup_transform.call_args.args[:2],('base_link','camera_color_optical_frame'))
        self.assertEqual(node.local_panel_pub.publish.call_count,1)
        node.pub.publish.assert_not_called()

    def test_local_depth_without_nonfuture_encoder_snapshot_is_not_published(self):
        node=ObservedDoorAngle.__new__(ObservedDoorAngle)
        node.info=object();node.rgbs=[object()]
        node.local_contact_stamp=10.2
        node.local_contact_anchor=np.array([.5,.2,.8])
        node.local_contact_history=deque([(9.,node.local_contact_anchor),(10.2,node.local_contact_anchor)])
        node.local_panel_pub=Mock();node.tf=Mock()
        msg=SimpleNamespace(header=SimpleNamespace(stamp=Time(seconds=10.1).to_msg()))
        node.publish_local_panel(msg)
        node.local_panel_pub.publish.assert_not_called()
        node.tf.lookup_transform.assert_not_called()

    def test_static_fit_excludes_entire_measured_plane_not_only_visible_patch(self):
        node,_=self.scan_fixture()
        panel=dict(valid=True,depth_stamp=10.,center=[1.,0.,.8],normal=[1.,0.,0.],
                   segment_xy=[[1.,0.],[1.,.11]],residual_m=.005,
                   orientation_uncertainty_rad=.01)
        node.exclusion_panel=lambda _:panel
        original=len(node.pending.ranges)
        ContactScanLocalization.process(node)
        cloud=node.reference.estimate.call_args.args[0]
        # The observed band extends beyond the short visible segment.
        self.assertLess(len(cloud),original)
        self.assertTrue(np.all(abs(cloud[:,0]-1.)>=.22))

    def test_delayed_tf_retries_original_encoder_anchor_without_resetting_angle_baseline(self):
        baseline=object()
        node=SimpleNamespace(local_contact_history=deque([(10.,np.array([.4,.1,.8]))]),
                             transform=Mock(),tracker=baseline)
        node.transform.side_effect=TransformException('pending scan')
        with self.assertRaisesRegex(ValueError,'no_fresh'):
            ObservedDoorAngle.retry_contact_anchor(node,10.1)
        node.transform.side_effect=None
        node.transform.return_value=(np.eye(3),np.array([1.,2.,0.]))
        result=ObservedDoorAngle.retry_contact_anchor(node,10.1)
        np.testing.assert_allclose(result,[1.4,2.1,.8])
        self.assertEqual(node.transform.call_args.args[0],'base_link')
        self.assertEqual(node.transform.call_args.args[1],Time(seconds=10.).to_msg())
        self.assertIs(node.tracker,baseline)

    def test_retry_anchor_rejects_old_future_and_missing_encoder_samples(self):
        for history in ([],[(9.,np.ones(3))],[(10.2,np.ones(3))]):
            node=SimpleNamespace(local_contact_history=history,transform=Mock())
            with self.assertRaisesRegex(ValueError,'no_fresh'):
                ObservedDoorAngle.retry_contact_anchor(node,10.1)
            node.transform.assert_not_called()

    def test_retry_can_use_older_still_fresh_sample_when_latest_tf_is_pending(self):
        node=SimpleNamespace(local_contact_history=[(10.,np.array([.4,0.,.8])),
            (10.1,np.array([.5,0.,.8]))],transform=Mock())
        node.transform.side_effect=[TransformException('pending'),(np.eye(3),np.zeros(3))]
        np.testing.assert_allclose(ObservedDoorAngle.retry_contact_anchor(node,10.15),[.4,0.,.8])
        self.assertEqual(node.transform.call_args.args[1],Time(seconds=10.).to_msg())

    def test_qualified_measured_long_line_fallback_publishes_yaw_only(self):
        node,msg=self.scan_fixture()
        node.reference.estimate.side_effect=ValueError('static scan correction discontinuity')
        node.yaw_reference=Mock()
        node.yaw_reference.estimate.return_value=dict(yaw=.01,yaw_uncertainty_rad=.005,
            residual_rms_m=.02,inliers=90,line_count=1,measured_spans_m=[2.5],
            translation_observed=False)
        node.exclusion_panel=lambda _:dict(valid=True,depth_stamp=10.,center=[2.,0.,.8],
            normal=[1.,0.,0.],residual_m=.005,orientation_uncertainty_rad=.01)
        ContactScanLocalization.process(node)
        node.publish.assert_called_once_with(msg.header.stamp)
        np.testing.assert_allclose(node.correction,[0.,0.,.01])
        q=node.report.call_args.kwargs
        self.assertFalse(q['full_pose_valid'])
        self.assertIsNone(q['translation_uncertainty_m'])
        self.assertEqual(q['yaw_source'],'frozen_observed_long_static_lines')

    def test_fresh_static_lines_break_missing_panel_cycle_without_icp_or_translation(self):
        node,msg=self.scan_fixture(translation_uncertainty=.01,translation_delta=.02)
        node.exclusion_panel=lambda _:None
        node.yaw_reference=Mock()
        node.yaw_reference.estimate.return_value=dict(yaw=.01,yaw_uncertainty_rad=.005,
            residual_rms_m=.02,inliers=90,line_count=1,measured_spans_m=[2.5],
            translation_observed=False)
        node.extension=Mock()
        ContactScanLocalization.process(node)
        node.reference.estimate.assert_not_called()
        node.extension.observe.assert_not_called()
        node.publish.assert_called_once_with(msg.header.stamp)
        np.testing.assert_allclose(node.correction,[0.,0.,.01])
        self.assertFalse(node.report.call_args.kwargs['full_pose_valid'])
        self.assertFalse(node.report.call_args.kwargs['fresh_panel_exclusion_used'])

    def test_missing_panel_and_unqualified_static_lines_do_not_publish_feedback(self):
        for reason in ('No continuous measured static yaw support',
                       'Independent measured static line headings disagree'):
            node,_=self.scan_fixture()
            node.exclusion_panel=lambda _:None
            node.yaw_reference=Mock()
            node.yaw_reference.estimate.side_effect=ValueError(reason)
            ContactScanLocalization.process(node)
            node.reference.estimate.assert_not_called()
            node.publish.assert_not_called()
            node.trust_scan.assert_not_called()
            self.assertFalse(node.scan_fit_valid)

    def test_stale_panel_only_allows_independent_static_yaw_not_point_fitting(self):
        node,_=self.scan_fixture()
        node.exclusion_panel=lambda _:dict(depth_stamp=9.)
        node.yaw_reference=Mock()
        node.yaw_reference.estimate.side_effect=ValueError('No continuous measured static yaw support')
        ContactScanLocalization.process(node)
        node.reference.estimate.assert_not_called()
        node.yaw_reference.estimate.assert_called_once()
        node.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
