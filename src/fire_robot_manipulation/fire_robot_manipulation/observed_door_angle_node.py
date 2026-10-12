"""Estimate door rotation using registered depth, observed handles and robot TF."""
from collections import deque
import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from cv_bridge import CvBridge
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_manipulation.observed_door_plane import fit_panel, PlaneRotation, rotation_matrix, fused_panel_segment
from fire_robot_manipulation.observed_panel_mask import panel_color_support
from fire_robot_manipulation.observed_lidar_panel import fit_lidar_panel


def seconds(header):
    return header.stamp.sec+header.stamp.nanosec*1.e-9


class ObservedDoorAngle(Node):
    def __init__(self):
        super().__init__('observed_door_angle')
        self.declare_parameter('observation_reference_frame', 'odom')
        self.reference_frame = self.get_parameter('observation_reference_frame').value
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.bridge = CvBridge()
        self.info = None
        self.rgbs = deque(maxlen=12)
        self.scans=deque(maxlen=5)
        self.anchor = None
        self.panel_color = None
        self.anchor_stamp = None
        self.contact_anchor=None
        self.contact_stamp=None
        self.local_contact_anchor=None
        self.local_contact_stamp=None
        self.local_contact_history=deque(maxlen=40)
        self.tracked_center = None
        self.tracked_stamp = None
        self.tracked_normal = None
        self.tracker = PlaneRotation()
        self.reference_yaw_history=deque(maxlen=32)
        self.baseline_yaw_uncertainty=None
        self.rng = np.random.default_rng(20260922)
        self.last_stamp = -1.
        self.pending_depth = None
        self.pub = self.create_publisher(JointState, '/observed_door/rotation', 10)
        self.status = self.create_publisher(String, '/observed_door/status', 10)
        self.local_panel_pub=self.create_publisher(String,'/observed_door/local_panel',10)
        self.sensor_callbacks=ReentrantCallbackGroup()
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.rgbs.append, qos_profile_sensor_data,
                                 callback_group=self.sensor_callbacks)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_cb, qos_profile_sensor_data,
                                 callback_group=self.sensor_callbacks)
        self.create_subscription(DoorInfo, '/detected_door', self.door_cb, 10)
        self.create_subscription(PointStamped,'/observed_door/contact_anchor',self.contact_cb,10)
        self.create_subscription(LaserScan,'/scan',self.scans.append,qos_profile_sensor_data,
                                 callback_group=self.sensor_callbacks)
        self.create_subscription(String, '/contact/session', self.session_cb, 10)
        if self.reference_frame=='contact_scan':
            self.create_subscription(String,'/contact/localization',self.localization_cb,10)
        self.create_timer(.02, self.process_depth)

    def session_cb(self, msg):
        # A new selected door needs a new observed closed-plane reference.
        self.anchor = self.anchor_stamp = self.panel_color = None
        self.contact_anchor = self.contact_stamp = None
        self.local_contact_anchor=self.local_contact_stamp=None
        self.local_contact_history.clear()
        self.tracked_center = self.tracked_stamp = self.tracked_normal = None
        self.tracker = PlaneRotation()
        self.baseline_yaw_uncertainty=None
        self.pending_depth = None
        self.report(False, self.get_clock().now().nanoseconds * 1e-9,
                    reason='new_target_baseline', target_id=msg.data)

    def localization_cb(self,msg):
        value=json.loads(msg.data)
        uncertainty=value.get('yaw_uncertainty_rad')
        stamp=value.get('measurement_stamp')
        if (not value.get('valid') or not isinstance(uncertainty,(int,float))
                or not isinstance(stamp,(int,float)) or not math.isfinite(uncertainty)
                or not math.isfinite(stamp) or not 0.<=uncertainty<=math.radians(2.)):
            return
        if value.get('event')=='static_reference_frozen':
            # Reset the still-closed baseline before grasp, in the measured
            # static frame that will also be used during panel rotation.
            self.tracker=PlaneRotation()
            self.baseline_yaw_uncertainty=None
            self.reference_yaw_history.clear()
        if value.get('event') in ('static_reference_frozen','static_scan_pose'):
            self.reference_yaw_history.append((stamp,uncertainty))

    def angle_uncertainty(self,stamp,panel_uncertainty):
        if self.reference_frame!='contact_scan':
            return panel_uncertainty
        eligible=[sample for sample in self.reference_yaw_history if 0.<=stamp-sample[0]<=.15]
        if not eligible:
            raise ValueError('no_timestamped_static_scan_yaw_confidence')
        current=max(eligible,key=lambda sample:sample[0])[1]
        baseline=(current if self.baseline_yaw_uncertainty is None
                  else self.baseline_yaw_uncertainty)
        # The camera plane-fit and LiDAR registration noise use independent
        # sensors (3-sigma confidence estimates). Keep both reference-frame yaw
        # errors fully correlated, then combine that group with camera noise.
        # This is a confidence estimate, not a deterministic error guarantee.
        bound=math.hypot(panel_uncertainty,current+baseline)
        if bound>math.radians(2.):
            raise ValueError('combined_panel_and_reference_yaw_uncertainty')
        if self.baseline_yaw_uncertainty is None:
            self.baseline_yaw_uncertainty=baseline
        return bound

    def publish_initial_geometry(self,fit,stamp,roi_source):
        # The static-reference initializer needs the measured panel mask before
        # an angle reference exists. This message cannot authorize completion.
        self.report(True,stamp,angle_measurement_valid=False,
            measurement_role='precontact_geometry_only_not_opening_completion',
            normal=fit['normal'].tolist(),center=fit['center'].tolist(),
            residual_m=fit['residual_m'],segment_xy=fit['segment_xy'].tolist(),
            segment_edge_margin_m=.05,segment_source='rgbd',inliers=fit['inliers'],
            orientation_uncertainty_rad=fit['orientation_uncertainty_rad'],
            roi_source=roi_source,sensor='rgbd')

    def info_cb(self, msg):
        self.info = msg

    def contact_cb(self,msg):
        if msg.header.frame_id=='base_link':
            point=np.array([msg.point.x,msg.point.y,msg.point.z])
            if np.all(np.isfinite(point)):
                self.local_contact_anchor=point
                self.local_contact_stamp=seconds(msg.header)
                self.local_contact_history.append((self.local_contact_stamp,point.copy()))
        try:
            rotation,translation=self.transform(msg.header.frame_id,msg.header.stamp)
            self.contact_anchor=rotation@np.array([msg.point.x,msg.point.y,msg.point.z])+translation
            self.contact_stamp=seconds(msg.header)
        except TransformException:
            return

    def transform(self, frame, stamp):
        tf = self.tf.lookup_transform(self.reference_frame, frame, Time.from_msg(stamp))
        q, t = tf.transform.rotation, tf.transform.translation
        return rotation_matrix([q.x,q.y,q.z,q.w]), np.array([t.x,t.y,t.z])

    def retry_contact_anchor(self, stamp):
        # An encoder callback may precede its scan TF. Retry the original
        # measurement, not a current/stale substitute, after TF becomes ready.
        eligible=[sample for sample in self.local_contact_history
                  if 0.<=stamp-sample[0]<=.25]
        for measured,point in reversed(eligible):
            try:
                rotation,translation=self.transform('base_link',Time(seconds=measured).to_msg())
                return rotation@point+translation
            except TransformException:
                continue
        raise ValueError('no_fresh_handle_or_continuous_panel_track')

    def door_cb(self, msg):
        if (msg.door_color != 'blue' or not msg.handle_detected
                or not msg.handle_detection_method.startswith(('yolo:primary', 'track:primary_yolo'))):
            return
        point = msg.handle_position
        if not point.header.frame_id:
            return
        try:
            rotation, translation = self.transform(point.header.frame_id, point.header.stamp)
            self.anchor = rotation@np.array([point.point.x,point.point.y,point.point.z])+translation
            self.anchor_stamp = seconds(point.header)
            self.panel_color = msg.door_color
        except (TransformException, ValueError):
            return

    def report(self, valid, stamp, **values):
        message = String()
        message.data = json.dumps(dict(valid=valid, depth_stamp=stamp,
            reference_frame=self.reference_frame,
            source='rgbd_or_lidar_panel_initialized_by_observed_handle', **values), allow_nan=False)
        self.status.publish(message)

    def depth_cb(self, msg):
        self.pending_depth = msg

    def try_lidar(self):
        scans=list(self.scans)
        if self.tracker.reference is None or self.tracked_center is None or not scans:
            return False
        latest=seconds(scans[-1].header)
        current=self.get_clock().now().nanoseconds*1.e-9
        if not -.02<=current-latest<=.15: return False
        clouds=[]; stamps=[]
        try:
            for scan in scans:
                stamp=seconds(scan.header)
                if latest-stamp>.18: continue
                ranges=np.asarray(scan.ranges,dtype=float)
                angles=scan.angle_min+np.arange(len(ranges))*scan.angle_increment
                valid=np.isfinite(ranges)&(ranges>scan.range_min)&(ranges<scan.range_max)
                xyz=np.column_stack((ranges[valid]*np.cos(angles[valid]),
                    ranges[valid]*np.sin(angles[valid]),np.zeros(int(valid.sum()))))
                rotation,translation=self.transform(scan.header.frame_id,scan.header.stamp)
                clouds.append(xyz@rotation.T+translation); stamps.append(stamp)
            if not clouds: return False
            # Prefer the newest scan. Accumulation is only needed when its
            # measured support cannot constrain the panel orientation.
            candidates=[(clouds[-1],stamps[-1],0.)]
            if len(clouds)>1:
                candidates.append((np.vstack(clouds),float(np.mean(stamps)),max(stamps)-min(stamps)))
            accepted=None
            for points,measured,window in candidates:
                if self.tracker.stamp is not None and measured<=self.tracker.stamp: continue
                try:
                    fit=fit_lidar_panel(points,self.tracked_center,self.tracked_normal,self.rng)
                    uncertainty=self.angle_uncertainty(measured,fit['orientation_uncertainty_rad'])
                    angle=self.tracker.update(fit['normal'],measured)
                    accepted=fit,angle,measured,window,uncertainty
                    break
                except (ValueError,np.linalg.LinAlgError):
                    continue
            if accepted is None: return False
            fit,angle,measured,window,uncertainty=accepted
            self.tracked_center=np.r_[fit['center'],self.tracked_center[2]]
            self.tracked_stamp=measured; self.tracked_normal=fit['normal']
            estimate=JointState(); estimate.header.stamp=Time(seconds=measured).to_msg()
            estimate.header.frame_id=self.reference_frame; estimate.name=['camera_door_rotation']
            estimate.position=[float(angle)]; self.pub.publish(estimate)
            self.report(True,measured,angle_rad=float(angle),normal=fit['normal'].tolist(),
                center=self.tracked_center.tolist(),residual_m=fit['residual_m'],
                plane_offset_uncertainty_m=fit['plane_offset_uncertainty_m'],
                segment_xy=fit['segment_xy'].tolist(),segment_edge_margin_m=fit['segment_edge_margin_m'],
                segment_source='visually_initialized_lidar',
                inliers=fit['inliers'],support_std_m=fit['support_std_m'],
                orientation_uncertainty_rad=uncertainty,
                panel_fit_uncertainty_rad=fit['orientation_uncertainty_rad'],
                roi_source='visually_initialized_lidar_panel',sensor='lidar',scan_window_sec=window)
            return True
        except (TransformException,ValueError,np.linalg.LinAlgError):
            return False

    def panel_segment_support(self, fit, stamp, angle):
        segment,margin,source=fit['segment_xy'],.05,'rgbd'
        candidates=[scan for scan in list(self.scans) if 0.<=stamp-seconds(scan.header)<=.15]
        if not candidates:
            return segment,margin,source
        try:
            candidates=sorted({seconds(scan.header):scan for scan in candidates}.items())
            clouds=[]
            for measured,scan in candidates:
                ranges=np.asarray(scan.ranges,float)
                angles=scan.angle_min+np.arange(len(ranges))*scan.angle_increment
                valid=np.isfinite(ranges)&(ranges>scan.range_min)&(ranges<scan.range_max)
                points=np.column_stack((ranges[valid]*np.cos(angles[valid]),
                    ranges[valid]*np.sin(angles[valid]),np.zeros(int(valid.sum()))))
                rotate,translation=self.transform(scan.header.frame_id,scan.header.stamp)
                clouds.append(points@rotate.T+translation)
            windows=[(clouds[-1],candidates[-1][0],0.)]
            if len(clouds)>1:
                windows.append((np.vstack(clouds),candidates[0][0],candidates[-1][0]-candidates[0][0]))
            for points,measured,window in windows:
                try:
                    lidar=fit_lidar_panel(points,fit['center'],fit['normal'],self.rng)
                    segment=fused_panel_segment(fit,lidar,angle,stamp-measured)
                    # The tracker bounds rotation at .45 rad/s. The fitted
                    # cloud is within .75m of the anchor; protect swept edges.
                    margin=lidar['segment_edge_margin_m']+(.035+.45*window)*.75 if window else lidar['segment_edge_margin_m']
                    source=('rgbd_with_accumulated_agreeing_lidar_support' if window
                        else 'rgbd_with_agreeing_lidar_support')
                    break
                except (ValueError,np.linalg.LinAlgError):
                    continue
        except (TransformException,ValueError,np.linalg.LinAlgError):
            pass
        return segment,margin,source

    def process_depth(self):
        msg = self.pending_depth
        if msg is None:
            return
        stamp = seconds(msg.header)
        if stamp-self.last_stamp < .09:
            return
        self.publish_local_panel(msg)
        try:
            if self.info is None or self.anchor is None or not self.rgbs:
                raise ValueError('waiting_for_calibration_and_observed_handle')
            rgb = min(list(self.rgbs), key=lambda image:abs(seconds(image.header)-stamp))
            if abs(seconds(rgb.header)-stamp) > .20:
                raise ValueError('no_recent_rgb_for_panel_segmentation')
            if self.tracked_stamp is not None and stamp-self.tracked_stamp <= .6:
                anchor, roi_source = self.tracked_center, 'tracked_depth_panel'
            elif self.anchor_stamp is not None and stamp-self.anchor_stamp <= 2.0:
                anchor, roi_source = self.anchor, 'recent_yolo_handle'
            elif self.contact_stamp is not None and -.02<=stamp-self.contact_stamp<=.25:
                anchor,roi_source=self.contact_anchor,'measured_grasp_fk_roi'
            else:
                anchor=self.retry_contact_anchor(stamp)
                roi_source='timestamped_grasp_fk_roi_retry'
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if msg.encoding == '16UC1':
                depth = depth.astype(np.float32)*.001
            elif msg.encoding != '32FC1':
                raise ValueError('unsupported_depth_encoding')
            if depth.shape != (self.info.height, self.info.width):
                raise ValueError('depth_not_registered_to_rgb_dimensions')
            v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
            z = depth[::4,::4]
            bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
            if bgr.shape[:2] != depth.shape:
                raise ValueError('color_depth_dimensions_differ')
            panel_mask = panel_color_support(bgr, self.panel_color)[::4,::4]
            valid = np.isfinite(z)&(z>.12)&(z<4.)&panel_mask
            z, u, v = z[valid], u[valid], v[valid]
            k = self.info.k
            points = np.column_stack(((u-k[2])*z/k[0], (v-k[5])*z/k[4], z))
            # RGB selects panel pixels; angles come from measured depth and TF.
            rotation, translation = self.transform(rgb.header.frame_id, msg.header.stamp)
            points = points@rotation.T+translation
            prior = ((self.tracked_normal, self.tracked_center)
                     if roi_source == 'tracked_depth_panel' else None)
            fit = fit_panel(points, anchor, self.rng, prior=prior)
            try:
                uncertainty=self.angle_uncertainty(stamp,fit['orientation_uncertainty_rad'])
            except ValueError:
                if self.tracker.reference is not None:
                    raise
                self.publish_initial_geometry(fit,stamp,roi_source)
                self.last_stamp=stamp
                return
            angle = self.tracker.update(fit['normal'], stamp)
            self.tracked_center, self.tracked_stamp = fit['center'], stamp
            self.tracked_normal = fit['normal']
            if angle is None:
                raise ValueError('collecting_closed_plane_baseline')
            estimate = JointState()
            estimate.header.stamp = msg.header.stamp
            estimate.header.frame_id = self.reference_frame
            estimate.name = ['camera_door_rotation']
            estimate.position = [float(angle)]
            self.pub.publish(estimate)
            segment,edge_margin,segment_source=self.panel_segment_support(fit,stamp,angle)
            self.report(True, stamp, angle_rad=float(angle), normal=fit['normal'].tolist(),
                        center=fit['center'].tolist(), residual_m=fit['residual_m'],
                        segment_xy=segment.tolist(),segment_edge_margin_m=edge_margin,
                        segment_source=segment_source,
                        inliers=fit['inliers'], inlier_ratio=fit['inlier_ratio'],
                        support_std_m=fit['support_std_m'],
                        orientation_uncertainty_rad=uncertainty,
                        panel_fit_uncertainty_rad=fit['orientation_uncertainty_rad'],
                        roi_source=roi_source,sensor='rgbd')
            self.last_stamp = stamp
        except TransformException as exc:
            # Keep the measurement briefly until its timestamped TF arrives.
            # Never substitute a newer pose or a simulator hinge measurement.
            age = self.get_clock().now().nanoseconds*1.e-9-stamp
            if age > .4:
                self.last_stamp = stamp
                self.report(False, stamp, reason=str(exc))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.last_stamp = stamp
            if not self.try_lidar():
                self.report(False, stamp, reason=str(exc))

    def publish_local_panel(self,msg):
        # Exclusion-only observation, not an angle completion certificate.
        # It depends on camera calibration and own wrist FK, not contact_scan TF.
        stamp=seconds(msg.header)
        history=list(getattr(self,'local_contact_history',()))
        if not history and self.local_contact_stamp is not None:
            history=[(self.local_contact_stamp,self.local_contact_anchor)]
        eligible=[sample for sample in history if 0.<=stamp-sample[0]<=.25]
        if (self.info is None or not eligible or not self.rgbs
                or stamp<=getattr(self,'last_local_panel_stamp',-1.)):
            return
        anchor_stamp,anchor=max(eligible,key=lambda sample:sample[0])
        self.last_local_panel_stamp=stamp
        try:
            rgb=min(list(self.rgbs),key=lambda image:abs(seconds(image.header)-stamp))
            if abs(seconds(rgb.header)-stamp)>.20:
                return
            depth=self.bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
            if msg.encoding=='16UC1':
                depth=depth.astype(np.float32)*.001
            elif msg.encoding!='32FC1':
                return
            bgr=self.bridge.imgmsg_to_cv2(rgb,desired_encoding='bgr8')
            if depth.shape!=(self.info.height,self.info.width) or bgr.shape[:2]!=depth.shape:
                return
            v,u=np.mgrid[0:depth.shape[0]:4,0:depth.shape[1]:4]
            z=depth[::4,::4]
            valid=np.isfinite(z)&(z>.12)&(z<4.)&panel_color_support(bgr,self.panel_color)[::4,::4]
            z,u,v=z[valid],u[valid],v[valid]
            k=self.info.k
            points=np.column_stack(((u-k[2])*z/k[0],(v-k[5])*z/k[4],z))
            tf=self.tf.lookup_transform('base_link',rgb.header.frame_id,Time.from_msg(msg.header.stamp))
            q,t=tf.transform.rotation,tf.transform.translation
            rotation=rotation_matrix([q.x,q.y,q.z,q.w])
            points=points@rotation.T+np.array([t.x,t.y,t.z])
            fit=fit_panel(points,anchor,self.rng)
            payload=dict(valid=True,depth_stamp=stamp,reference_frame='base_link',
                source='registered_rgbd_and_own_wrist_fk_exclusion_only',sensor='rgbd',
                center=fit['center'].tolist(),normal=fit['normal'].tolist(),
                segment_xy=fit['segment_xy'].tolist(),residual_m=fit['residual_m'],
                orientation_uncertainty_rad=fit['orientation_uncertainty_rad'],
                inliers=fit['inliers'],anchor_measurement_stamp=anchor_stamp)
            self.local_panel_pub.publish(String(data=json.dumps(payload,allow_nan=False)))
        except (ValueError,np.linalg.LinAlgError,TransformException):
            return


def main(args=None):
    rclpy.init(args=args)
    node = ObservedDoorAngle()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
