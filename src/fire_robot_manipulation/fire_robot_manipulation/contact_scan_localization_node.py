"""Contact-only localization against frozen, observed static LiDAR returns.

The new child frame does not replace Nav2 odometry or SLAM. Environment model
states, joint truth and world geometry are not subscribed to or loaded here.
"""
from collections import deque
import json
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformException

from .contact_scan_registration import (
    StaticScanRegistration, ScanPose, CorrectionFilter, transform_cloud, compose_pose, inverse_pose,
    exclude_observed_plane, temporal_static_reference,
    ObservedStaticExtension)
from .static_line_yaw import StaticLineYaw


def planar_transform(transform):
    t, q = transform.translation, transform.rotation
    yaw = math.atan2(2.*(q.w*q.z+q.x*q.y), 1.-2.*(q.y*q.y+q.z*q.z))
    return np.array([t.x, t.y, yaw])


class ContactScanLocalization(Node):
    def __init__(self):
        super().__init__('contact_scan_localization')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.broadcaster = TransformBroadcaster(self)
        self.status = self.create_publisher(String, '/contact/localization', 10)
        self.correction = np.zeros(3)
        self.filter = CorrectionFilter(self.correction,yaw_time_constant=.4)
        self.reference = None
        self.original_reference = None
        self.yaw_reference = None
        self.extension = None
        self.static_scans = deque(maxlen=32)
        self.panel = None
        self.local_panel = None
        self.pending = None
        self.pending_wheel = None
        self.freeze_requested = False
        self.phase = 'WAITING'
        self.last_stamp = -1.
        self.trusted_scan_stamp = None
        self.trusted_scan_received = None
        self.scan_fit_valid = False
        self.trusted_scan_quality = {}
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.wheel_cb, qos_profile_sensor_data)
        self.create_subscription(String, '/observed_door/status', self.panel_cb, 10)
        self.create_subscription(String,'/observed_door/local_panel',self.local_panel_cb,10)
        self.create_subscription(String, '/manipulation_phase', self.phase_cb, 10)
        self.create_timer(.01, self.process)

    def scan_cb(self, msg):
        self.pending = msg

    def wheel_cb(self, msg):
        self.pending_wheel = msg
        self.process_wheel()

    def process_wheel(self):
        # This launch maps wheel-integrated odometry to /odom. Propagate an
        # accepted, still-fresh scan correction with actual wheel timestamps;
        # do not pretend that a new scan was measured at the wheel timestamp.
        msg = self.pending_wheel
        if msg is None:
            return
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
        now = self.get_clock().now().nanoseconds*1.e-9
        # Clock and odometry callbacks may arrive in either order. Wait for
        # the same wheel timestamp instead of extrapolating or retimestamping.
        if 0. < stamp-now <= .15:
            return
        self.pending_wheel = None
        if (msg.header.frame_id != 'odom' or msg.child_frame_id not in ('base_footprint', 'base_link')
                or not self.scan_fit_valid or self.trusted_scan_stamp is None
                or self.trusted_scan_received is None
                or not 0. <= stamp-self.trusted_scan_stamp <= .15
                or not 0. <= now-stamp <= .15
                or not 0. <= time.monotonic()-self.trusted_scan_received <= 1.):
            return
        self.publish(msg.header.stamp)
        if stamp-getattr(self, 'last_propagation_report', -1.) >= 1.:
            self.last_propagation_report = stamp
            self.report(self.trusted_scan_stamp, True, event='wheel_pose_propagated',
                        pose_stamp=stamp, correction_measurement_age_sec=stamp-self.trusted_scan_stamp,
                        source='wheel_encoders_with_fresh_observed_scan_correction',
                        **getattr(self, 'trusted_scan_quality', {}))

    def trust_scan(self, stamp):
        self.trusted_scan_stamp = stamp
        self.trusted_scan_received = time.monotonic()
        self.scan_fit_valid = True

    def panel_cb(self, msg):
        value = json.loads(msg.data)
        if (value.get('valid') and value.get('reference_frame') == 'contact_scan'
                and 'segment_xy' in value):
            self.panel = value

    def local_panel_cb(self,msg):
        value=json.loads(msg.data)
        if (value.get('valid') and value.get('reference_frame')=='base_link'
                and value.get('source')=='registered_rgbd_and_own_wrist_fk_exclusion_only'):
            self.local_panel=value

    def exclusion_panel(self,stamp):
        panel=self.local_panel
        if panel is not None and 0.<=stamp-panel['depth_stamp']<=.25:
            try:
                wheel=planar_transform(self.tf.lookup_transform(
                    'odom','base_link',Time(seconds=float(panel['depth_stamp']))).transform)
                pose=compose_pose(self.correction,wheel)
                c,s=math.cos(pose[2]),math.sin(pose[2])
                rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
                transformed=dict(panel)
                transformed['center']=(rotation@np.asarray(panel['center'])+np.r_[pose[:2],0.]).tolist()
                transformed['normal']=(rotation@np.asarray(panel['normal'])).tolist()
                transformed['segment_xy']=transform_cloud(panel['segment_xy'],pose).tolist()
                transformed['reference_frame']='contact_scan'
                return transformed
            except (TransformException,ValueError,KeyError):
                pass
        return self.panel

    def phase_cb(self, msg):
        phase = msg.data.split(':', 1)[0]
        self.phase = phase
        if phase in ('LOCALIZE_HANDLE', 'COMPLETE') or phase.startswith('FAILED_'):
            # Keep the correction continuous across jobs. Only the observed
            # static reference is refreshed while no door is being manipulated.
            self.reference = None
            self.original_reference = None
            self.yaw_reference = None
            self.extension = None
            self.local_panel = None
            self.freeze_requested = False
            self.filter = CorrectionFilter(self.correction,yaw_time_constant=.4)
            self.scan_fit_valid = False
            self.trusted_scan_quality = {}
            if phase=='LOCALIZE_HANDLE':
                # The adapter has already stopped at its observed docking pose.
                # Build the closed static reference before the initial angle gate.
                self.freeze_requested=True
        elif phase == 'PRE_GRASP':
            self.freeze_requested = True

    def report(self, stamp, valid, **values):
        self.status.publish(String(data=json.dumps(dict(
            valid=valid, measurement_stamp=stamp, frame='contact_scan',
            environment_truth_used=False, **values), allow_nan=False)))

    def publish(self, stamp):
        # odom already has a parent (map). Publish a new child instead of
        # giving odom a conflicting second parent.
        pose = inverse_pose(self.correction)
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'contact_scan'
        msg.transform.translation.x, msg.transform.translation.y = map(float, pose[:2])
        msg.transform.rotation.z = math.sin(pose[2]/2.)
        msg.transform.rotation.w = math.cos(pose[2]/2.)
        self.broadcaster.sendTransform(msg)

    def process(self):
        self.process_wheel()
        msg = self.pending
        if msg is None:
            return
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
        now = self.get_clock().now().nanoseconds*1.e-9
        if stamp <= self.last_stamp or not -.02 <= now-stamp <= .15:
            self.pending = None
            return
        try:
            wheel = planar_transform(self.tf.lookup_transform(
                'odom', msg.header.frame_id, Time.from_msg(msg.header.stamp)).transform)
        except TransformException:
            return
        self.pending = None
        self.last_stamp = stamp
        ranges = np.asarray(msg.ranges, float)
        angles = msg.angle_min+np.arange(len(ranges))*msg.angle_increment
        selected = np.isfinite(ranges)&(ranges > max(.4, msg.range_min))&(ranges < min(8., msg.range_max))
        cloud = np.column_stack((ranges[selected]*np.cos(angles[selected]),
                                 ranges[selected]*np.sin(angles[selected])))
        seed = compose_pose(self.correction, wheel)
        observed = transform_cloud(cloud, seed)
        panel = self.exclusion_panel(stamp)
        fresh_panel = panel is not None and -.02 <= stamp-panel['depth_stamp'] <= .5
        try:
            if fresh_panel:
                # A visible RGB-D patch is not the physical edge of the door.
                # Discard its whole measured plane band from static fitting.
                observed = exclude_observed_plane(observed, panel)
            if self.reference is None:
                rays=np.full((len(ranges),2),np.nan)
                rays[selected]=transform_cloud(cloud,seed)
                self.static_scans.append((stamp,seed.copy(),rays))
                if self.freeze_requested:
                    if self.phase not in ('LOCALIZE_HANDLE','PRE_GRASP'):
                        self.scan_fit_valid=False
                        self.report(stamp,False,reason='static_reference_not_ready_before_contact')
                        return
                    if not fresh_panel:
                        self.scan_fit_valid = False
                        self.report(stamp, False, reason='awaiting_static_reference_and_visual_panel')
                        return
                    try:
                        points=temporal_static_reference(list(self.static_scans))
                    except ValueError as exc:
                        # Keep original wheel timestamps while collecting the
                        # stationary, pre-contact sensor baseline.
                        self.trust_scan(stamp)
                        self.publish(msg.header.stamp)
                        self.report(stamp,False,event='collecting_stationary_reference',reason=str(exc))
                        return
                    points = exclude_observed_plane(points, panel)
                    # A measured voxel grid bounds fitting cost without adding
                    # any unseen geometry to the reference.
                    _, keep = np.unique(np.floor(points/.015).astype(int), axis=0, return_index=True)
                    self.reference = StaticScanRegistration(points[keep])
                    self.original_reference = self.reference
                    self.yaw_reference = StaticLineYaw(points[keep])
                    self.extension = ObservedStaticExtension(points[keep])
                    self.trusted_scan_quality=dict(full_pose_valid=False,
                        translation_source='stationary_own_encoder_reference',
                        translation_uncertainty_m=None,yaw_uncertainty_rad=math.radians(.2))
                    self.report(stamp, True, event='static_reference_frozen',
                                measured_points=len(points[keep]), visual_panel_excluded=True,
                                basis='stationary_original_beam_temporal_median',
                                observation_window_sec=2.,minimum_independent_scans=17,
                                observed_reference_points_xy=points[keep].tolist(),
                                **self.trusted_scan_quality)
                self.trust_scan(stamp)
                self.publish(msg.header.stamp)
                return
            # Filter in the seed frame, then restore the actual laser frame.
            cloud = transform_cloud(observed, inverse_pose(seed))
            yaw_only_fit=None
            try:
                if not fresh_panel:
                    raise ValueError('Point fitting requires a fresh moving panel exclusion')
                estimate = self.reference.estimate(cloud, seed,require_full_translation=False)
                delta_yaw=math.atan2(math.sin(estimate.pose[2]-seed[2]),
                                    math.cos(estimate.pose[2]-seed[2]))
                if abs(delta_yaw)>math.radians(2.):
                    raise ValueError('static scan correction discontinuity')
            except (ValueError,np.linalg.LinAlgError):
                # Long, freshly measured static lines constrain yaw even while
                # the camera loses the panel. This breaks the panel/TF wait
                # cycle without permitting unmasked ICP or new map anchors.
                yaw_reference=getattr(self,'yaw_reference',None)
                if yaw_reference is None:
                    raise
                yaw_only_fit=yaw_reference.estimate(cloud,seed[2])
                estimate=ScanPose(np.r_[seed[:2],yaw_only_fit['yaw']],
                    yaw_only_fit['residual_rms_m'],None,
                    yaw_only_fit['yaw_uncertainty_rad'],yaw_only_fit['inliers'])
            # New regions are admitted only from independently qualified fits
            # against the original anchors, not from an accumulating map loop.
            extension=getattr(self,'extension',None)
            if (yaw_only_fit is None and extension is not None and fresh_panel
                    and 0.<=stamp-panel['depth_stamp']<=.25):
                try:
                    anchor=(estimate if self.reference is self.original_reference else
                            self.original_reference.estimate(cloud,seed))
                    anchor_delta=anchor.pose-seed
                    anchor_delta[2]=math.atan2(math.sin(anchor_delta[2]),math.cos(anchor_delta[2]))
                    if (np.linalg.norm(anchor_delta[:2])<=.05
                            and abs(anchor_delta[2])<=math.radians(2.)):
                        shift=compose_pose(anchor.pose,inverse_pose(seed))
                        rotation=np.array([[math.cos(shift[2]),-math.sin(shift[2]),0.],
                                           [math.sin(shift[2]), math.cos(shift[2]),0.],
                                           [0.,0.,1.]])
                        corrected_panel=dict(panel)
                        corrected_panel['center']=(rotation@np.asarray(panel['center'])+np.r_[shift[:2],0.]).tolist()
                        corrected_panel['normal']=(rotation@np.asarray(panel['normal'])).tolist()
                        points=exclude_observed_plane(transform_cloud(cloud,anchor.pose),corrected_panel)
                        update=extension.observe(stamp,points,
                            translation_uncertainty_m=anchor.translation_uncertainty_m,
                            yaw_uncertainty_rad=anchor.yaw_uncertainty_rad)
                        if update is not None:
                            self.reference,provenance=update
                            self.report(stamp,True,event='observed_static_reference_extended',**provenance)
                except (ValueError,np.linalg.LinAlgError):
                    # An uncertain new region must not alter old anchors or the
                    # currently qualified correction.
                    pass
            delta = estimate.pose-seed
            delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
            full_pose=bool(estimate.translation_uncertainty_m is not None
                           and estimate.translation_uncertainty_m<=.025
                           and np.linalg.norm(delta[:2])<=.05)
            if abs(delta[2]) > math.radians(2.):
                raise ValueError('static scan correction discontinuity')
            # Check the candidate before committing it or publishing fresh TF.
            candidate_filter=CorrectionFilter(self.filter.pose,self.filter.time_constant,
                yaw_time_constant=self.filter.yaw_time_constant)
            candidate_filter.stamp=self.filter.stamp
            candidate_correction=candidate_filter.update_at_sensor_pivot(
                estimate.pose,wheel,stamp,apply_translation=full_pose)
            filtered_pose = compose_pose(candidate_correction, wheel)
            lag = np.linalg.norm(filtered_pose[:2]-estimate.pose[:2])
            yaw_lag=abs(math.atan2(math.sin(filtered_pose[2]-estimate.pose[2]),
                                  math.cos(filtered_pose[2]-estimate.pose[2])))
            extension_yaw=0. if extension is None or yaw_only_fit is not None else extension.yaw_uncertainty_rad
            yaw_uncertainty=math.hypot(math.radians(.2),estimate.yaw_uncertainty_rad)+extension_yaw+yaw_lag
            if yaw_uncertainty>math.radians(2.):
                raise ValueError('Filtered static scan yaw has unbounded measurement uncertainty')
            self.filter=candidate_filter
            self.correction=candidate_correction
            quality=dict(full_pose_valid=full_pose,
                fresh_panel_exclusion_used=bool(fresh_panel),
                yaw_source=('frozen_observed_long_static_lines' if yaw_only_fit is not None
                            else 'static_point_to_line_registration'),
                independent_line_support=yaw_only_fit,
                translation_source=('qualified_static_scan_and_wheel_encoders' if full_pose
                    else 'wheel_encoders_with_yaw_correction_about_current_sensor'),
                translation_uncertainty_m=(math.hypot(.015, estimate.translation_uncertainty_m)+float(lag)
                    if full_pose else None),
                unapplied_scan_translation_uncertainty_m=(None if full_pose
                    else estimate.translation_uncertainty_m),
                unapplied_scan_translation_delta_m=(None if full_pose
                    else float(np.linalg.norm(delta[:2]))),
                yaw_uncertainty_rad=yaw_uncertainty,
                yaw_fit_uncertainty_rad=estimate.yaw_uncertainty_rad,
                extension_anchor_yaw_uncertainty_rad=extension_yaw,
                yaw_filter_lag_rad=yaw_lag,
                yaw_filter_time_constant_sec=self.filter.yaw_time_constant)
            self.trusted_scan_quality=quality
            self.trust_scan(stamp)
            self.publish(msg.header.stamp)
            self.report(stamp, True, event='static_scan_pose', correction=self.correction.tolist(),
                residual_rms_m=estimate.residual_rms_m, inliers=estimate.inliers,
                # Include reference/calibration uncertainty, not just fit covariance.
                **quality)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Never turn a failed fit into freshly stamped wheel-only feedback.
            self.scan_fit_valid = False
            self.report(stamp, False, reason=str(exc))


def main(args=None):
    rclpy.init(args=args)
    node = ContactScanLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
