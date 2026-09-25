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
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from cv_bridge import CvBridge
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_manipulation.observed_door_plane import fit_panel, PlaneRotation, rotation_matrix
from fire_robot_manipulation.observed_panel_mask import panel_color_support
from fire_robot_manipulation.observed_lidar_panel import fit_lidar_panel


def seconds(header):
    return header.stamp.sec+header.stamp.nanosec*1.e-9


class ObservedDoorAngle(Node):
    def __init__(self):
        super().__init__('observed_door_angle')
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
        self.tracked_center = None
        self.tracked_stamp = None
        self.tracked_normal = None
        self.tracker = PlaneRotation()
        self.rng = np.random.default_rng(20260922)
        self.last_stamp = -1.
        self.pending_depth = None
        self.pub = self.create_publisher(JointState, '/observed_door/rotation', 10)
        self.status = self.create_publisher(String, '/observed_door/status', 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.rgbs.append, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(DoorInfo, '/detected_door', self.door_cb, 10)
        self.create_subscription(PointStamped,'/observed_door/contact_anchor',self.contact_cb,10)
        self.create_subscription(LaserScan,'/scan',self.scans.append,qos_profile_sensor_data)
        self.create_timer(.02, self.process_depth)

    def info_cb(self, msg):
        self.info = msg

    def contact_cb(self,msg):
        try:
            rotation,translation=self.transform(msg.header.frame_id,msg.header.stamp)
            self.contact_anchor=rotation@np.array([msg.point.x,msg.point.y,msg.point.z])+translation
            self.contact_stamp=seconds(msg.header)
        except TransformException:
            return

    def transform(self, frame, stamp):
        tf = self.tf.lookup_transform('odom', frame, Time.from_msg(stamp))
        q, t = tf.transform.rotation, tf.transform.translation
        return rotation_matrix([q.x,q.y,q.z,q.w]), np.array([t.x,t.y,t.z])

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
            source='rgbd_or_lidar_panel_initialized_by_observed_handle', **values), allow_nan=False)
        self.status.publish(message)

    def depth_cb(self, msg):
        self.pending_depth = msg

    def try_lidar(self):
        if self.tracker.reference is None or self.tracked_center is None or not self.scans:
            return False
        latest=seconds(self.scans[-1].header)
        current=self.get_clock().now().nanoseconds*1.e-9
        if not -.02<=current-latest<=.15: return False
        clouds=[]; stamps=[]
        try:
            for scan in list(self.scans):
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
                    angle=self.tracker.update(fit['normal'],measured)
                    accepted=fit,angle,measured,window
                    break
                except (ValueError,np.linalg.LinAlgError):
                    continue
            if accepted is None: return False
            fit,angle,measured,window=accepted
            self.tracked_center=np.r_[fit['center'],self.tracked_center[2]]
            self.tracked_stamp=measured; self.tracked_normal=fit['normal']
            estimate=JointState(); estimate.header.stamp=Time(seconds=measured).to_msg()
            estimate.header.frame_id='odom'; estimate.name=['camera_door_rotation']
            estimate.position=[float(angle)]; self.pub.publish(estimate)
            self.report(True,measured,angle_rad=float(angle),normal=fit['normal'].tolist(),
                center=self.tracked_center.tolist(),residual_m=fit['residual_m'],
                inliers=fit['inliers'],support_std_m=fit['support_std_m'],
                orientation_uncertainty_rad=fit['orientation_uncertainty_rad'],
                roi_source='visually_initialized_lidar_panel',sensor='lidar',scan_window_sec=window)
            return True
        except (TransformException,ValueError,np.linalg.LinAlgError):
            return False

    def process_depth(self):
        msg = self.pending_depth
        if msg is None:
            return
        stamp = seconds(msg.header)
        if stamp-self.last_stamp < .09:
            return
        try:
            if self.info is None or self.anchor is None or not self.rgbs:
                raise ValueError('waiting_for_calibration_and_observed_handle')
            rgb = min(self.rgbs, key=lambda image:abs(seconds(image.header)-stamp))
            if abs(seconds(rgb.header)-stamp) > .20:
                raise ValueError('no_recent_rgb_for_panel_segmentation')
            if self.tracked_stamp is not None and stamp-self.tracked_stamp <= .6:
                anchor, roi_source = self.tracked_center, 'tracked_depth_panel'
            elif self.anchor_stamp is not None and stamp-self.anchor_stamp <= 2.0:
                anchor, roi_source = self.anchor, 'recent_yolo_handle'
            elif self.contact_stamp is not None and -.02<=stamp-self.contact_stamp<=.25:
                anchor,roi_source=self.contact_anchor,'measured_grasp_fk_roi'
            else:
                raise ValueError('no_fresh_handle_or_continuous_panel_track')
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
            angle = self.tracker.update(fit['normal'], stamp)
            self.tracked_center, self.tracked_stamp = fit['center'], stamp
            self.tracked_normal = fit['normal']
            if angle is None:
                raise ValueError('collecting_closed_plane_baseline')
            estimate = JointState()
            estimate.header.stamp = msg.header.stamp
            estimate.header.frame_id = 'odom'
            estimate.name = ['camera_door_rotation']
            estimate.position = [float(angle)]
            self.pub.publish(estimate)
            self.report(True, stamp, angle_rad=float(angle), normal=fit['normal'].tolist(),
                        center=fit['center'].tolist(), residual_m=fit['residual_m'],
                        inliers=fit['inliers'], inlier_ratio=fit['inlier_ratio'],
                        support_std_m=fit['support_std_m'],
                        orientation_uncertainty_rad=fit['orientation_uncertainty_rad'],roi_source=roi_source,sensor='rgbd')
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
