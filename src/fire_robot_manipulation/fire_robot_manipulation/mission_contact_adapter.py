"""Reconnect mission completion to live, reachable, selected-door contact work."""
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import numpy as np
import rclpy
import tf2_geometry_msgs
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_interfaces.srv import OpenDoor
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import Float64, String
from tf2_ros import Buffer, TransformListener, TransformException
from fire_robot_navigation.observed_footprint import ObservedFootprintGuard
from .piper_actual_kinematics import (
    PiperActualKinematics, PIPER_JOINT_LIMITS, PIPER_STOW, PIPER_BODY_OUTLINE, _rot_z)
from .contact_feedback import observed_panel_docking_lanes, SimMotionDeadline, observed_clearance
from .confirmed_handle_track import ConfirmedHandleTrack


# The supervisor must outlast the inner 3000 s wall budget plus release/stow.
# Sensor freshness, measured-progress and simulation-time guards stay inside.
CONTACT_OPERATION_WALL_TIMEOUT_SEC = 3600.


def panel_view_command(point, camera_xyz, optical_yaw):
    delta = np.asarray(point, float)[:2]-np.asarray(camera_xyz, float)[:2]
    if not np.isfinite(delta).all() or not math.isfinite(optical_yaw):
        raise ValueError('Invalid observed panel view')
    if np.linalg.norm(delta) < .30:
        raise ValueError('Panel too close for view acquisition')
    bearing = math.atan2(delta[1], delta[0])-optical_yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    return ((-.04, 0.) if abs(bearing) < .04
            else (0., float(np.clip(.8*bearing, -.12, .12))))


class MissionContactAdapter(Node):
    def __init__(self):
        super().__init__('mission_contact_adapter')
        self.declare_parameter('contact_reference_frame', 'odom')
        self.reference_frame = self.get_parameter('contact_reference_frame').value
        group = ReentrantCallbackGroup()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.client = self.create_client(OpenDoor, '/physical_open_door', callback_group=group)
        self.create_service(OpenDoor, '/open_door', self.open, callback_group=group)
        self.create_subscription(DoorInfo, '/detected_door', self.observation, 20, callback_group=group)
        self.create_subscription(DoorInfo, '/contact/handle_observation',
                                 lambda m: self.observation(m, handle_only=True), 10, callback_group=group)
        self.create_subscription(LaserScan, '/scan', self.scan_observation, 5,
                                 callback_group=group)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info,
                                 qos_profile_sensor_data, callback_group=group)
        self.create_subscription(String, '/contact/handle_geometry', self.handle_geometry,
                                 10, callback_group=group)
        self.create_subscription(String, '/observed_door/status', self.panel_observation, 10,
                                 callback_group=group)
        self.cmd = self.create_publisher(Twist, '/cmd_vel_manual', 10)
        self.guarded_cmd = self.create_publisher(Twist, '/cmd_vel_manual_guarded', 10)
        self.return_pose_pub = self.create_publisher(PoseStamped, '/contact/return_pose', 10)
        self.view_guard = ObservedFootprintGuard(self)
        self.selected = self.create_publisher(DoorInfo, '/contact/target_observation', 10)
        self.session = self.create_publisher(String, '/contact/session', 10)
        self.events = self.create_publisher(String, '/contact/mission_events', 20)
        self.arm = [self.create_publisher(Float64, f'/fire_robot/sim_arm/joint{i}_cmd', 10)
                    for i in range(1, 7)]
        self.began = time.monotonic()
        self.active = False
        self.request_target = None
        self.latest = None
        self.view_hint = None
        self.handle_track=ConfirmedHandleTrack(acquire=.45)
        self.scan = None
        self.scan_received = 0.
        self.panel = None
        self.camera_bearing_limit = None
        self.grip_lane_y = None
        self.handle_spans = deque(maxlen=12)
        self.handle_positions = deque(maxlen=12)
        self.camera_xyz = None
        self.docking_posture = None
        self.sequence = 0
        self.kinematics = PiperActualKinematics()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.create_timer(.25, self.initial_stow)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def initial_stow(self):
        if time.monotonic() - self.began < 20 and not self.active:
            for pub, value in zip(self.arm, PIPER_STOW):
                pub.publish(Float64(data=float(value)))

    def event(self, name, **values):
        payload = dict(event=name, sim_time=self.now(), session=self.sequence,
                       contact_reference_frame=self.reference_frame, **values)
        self.events.publish(String(data=json.dumps(payload, allow_nan=False)))
        self.get_logger().info('MISSION_CONTACT ' + json.dumps(payload))

    def transform(self, point, frame):
        transform = self.tf.lookup_transform(frame, point.header.frame_id,
                                            Time.from_msg(point.header.stamp))
        return tf2_geometry_msgs.do_transform_point(point, transform)

    def observation(self, msg, handle_only=False):
        if not self.active or self.request_target is None:
            return
        if ((not handle_only and msg.door_color != 'blue') or not msg.handle_detected
                or not msg.handle_detection_method.startswith(('yolo:primary', 'track:primary_yolo'))
                or not msg.handle_detection_method.endswith(':registered_depth')
                or not math.isfinite(msg.handle_confidence)):
            return
        track=getattr(self,'handle_track',None)
        if track is None:
            track=self.handle_track=ConfirmedHandleTrack(acquire=.45)
        if msg.handle_confidence < track.minimum:
            return
        try:
            base = self.transform(msg.handle_position, 'base_link')
            mapped = self.transform(msg.handle_position, self.reference_frame)
        except TransformException:
            return
        p, m = base.point, mapped.point
        target = self.request_target.point
        if math.hypot(m.x-target.x, m.y-target.y) > .90:
            return
        if not (.30 < p.x < 2.0 and abs(p.y) < .60 and .45 < p.z < 1.05):
            return
        stamp = base.header.stamp.sec + base.header.stamp.nanosec * 1e-9
        if not (-.1 < self.now()-stamp < .8):
            return
        # A weak detection may improve the camera view, never authorize grasp.
        hint = getattr(self, 'view_hint', None)
        if hint is None or stamp > hint[0]:
            self.view_hint = (stamp, np.array([m.x, m.y, m.z]), msg.handle_confidence)
        if not track.update([m.x,m.y,m.z],msg.handle_confidence,stamp):
            return
        if not self.handle_positions or stamp>self.handle_positions[-1][0]:
            self.handle_positions.append((stamp,np.array([m.x,m.y,m.z])))
        selected = copy.deepcopy(msg)
        selected.door_color = 'blue'
        selected.handle_position = base
        self.latest = (selected, stamp, np.array([m.x, m.y, m.z]))
        self.selected.publish(selected)

    def current_handle_xyz(self, latest):
        # Retain the exposure timestamp; only the coordinate frame is updated
        # using current robot odometry. This does not create a new observation.
        observed, exposure, anchored = latest
        now=self.now()
        if not 0. <= now-exposure < .8:
            raise ValueError('Original handle exposure is stale or future')
        transform=self.tf.lookup_transform('base_link',self.reference_frame,Time())
        stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
        if not 0. <= now-stamp <= .15:
            raise ValueError('Current robot transform is stale or future')
        point=copy.deepcopy(observed.handle_position)
        point.header.frame_id=self.reference_frame
        point.point.x,point.point.y,point.point.z=map(float,anchored)
        current=tf2_geometry_msgs.do_transform_point(point,transform).point
        xyz=np.array([current.x,current.y,current.z])
        if not np.all(np.isfinite(xyz)):
            raise ValueError('Nonfinite current handle coordinates')
        return xyz

    def command(self, forward=0., turn=0.):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(forward), float(turn)
        self.cmd.publish(msg)

    def scan_observation(self, msg):
        self.scan, self.scan_received = msg, self.now()

    def guarded_view_command(self, forward, turn):
        scene = self.view_guard.snapshot(self.scan, self.scan_received)
        msg = Twist()
        if scene is not None and scene.safe(forward, turn):
            msg.linear.x, msg.angular.z = forward, turn
        self.guarded_cmd.publish(msg)
        return msg.linear.x, msg.angular.z

    def reobserve_docking_support(self):
        deadline = SimMotionDeadline(self.now(), time.monotonic(), 15.,
                                     stall_limit=3., wall_limit=180.)
        stamps, report = set(), -1.
        while rclpy.ok() and not deadline.failure(self.now(), time.monotonic()):
            panel, latest = self.panel, self.latest
            if (panel is None or latest is None or not self.handle_spans
                    or self.camera_bearing_limit is None
                    or not 0. <= self.now()-panel.get('depth_stamp', -1.) <= .6
                    or not 0. <= self.now()-latest[1] <= .8):
                candidate, applied, basis = self.reobserve_selected_view()
                if self.now()-report > 1.:
                    report = self.now()
                    self.event('reobserve_selected_view', candidate=list(candidate),
                        applied=list(applied), basis=basis,
                        primary_confidence_gate=self.handle_track.acquire,
                        grasp_authorized=False)
                time.sleep(.05)
                continue
            try:
                if panel.get('reference_frame') != self.reference_frame:
                    raise ValueError('No time-aligned observed panel span')
                anchor = latest[2][:2]
                normal = np.asarray(panel['normal'][:2], float)
                normal /= np.linalg.norm(normal)
                segment = np.asarray(panel['segment_xy'], float)
                margin = panel.get('segment_edge_margin_m', .05)
                try:
                    observed_panel_docking_lanes(anchor, normal, segment, PIPER_BODY_OUTLINE, margin)
                except ValueError as exc:
                    if 'too narrow' not in str(exc):
                        raise
                    stamps.clear()
                else:
                    self.guarded_view_command(0., 0.)
                    stamps.add(panel['depth_stamp'])
                    if len(stamps) >= 3:
                        self.event('observed_panel_view_ready', distinct_frames=len(stamps),
                            measured_span_m=float(np.linalg.norm(segment[1]-segment[0])),
                            simulator_geometry_used=False)
                        return True
                    time.sleep(.05)
                    continue
                tf = self.tf.lookup_transform('base_link', self.reference_frame, Time())
                stamp = tf.header.stamp.sec+tf.header.stamp.nanosec*1.e-9
                if not 0. <= self.now()-stamp <= .15:
                    raise ValueError('Stale current robot transform')
                point = PointStamped()
                point.header.frame_id = self.reference_frame
                point.point.x, point.point.y = map(float, segment.mean(axis=0))
                midpoint = tf2_geometry_msgs.do_transform_point(point, tf).point
                camera = self.tf.lookup_transform('base_link', 'camera_color_optical_frame', Time()).transform
                q, c = camera.rotation, camera.translation
                optical_yaw = math.atan2(2*(q.y*q.z-q.w*q.x), 2*(q.x*q.z+q.w*q.y))
                candidate = panel_view_command([midpoint.x, midpoint.y], [c.x, c.y], optical_yaw)
                applied = self.guarded_view_command(*candidate)
                if self.now()-report > 1.:
                    report = self.now()
                    self.event('reobserve_partial_panel',
                        measured_span_m=float(np.linalg.norm(segment[1]-segment[0])),
                        required_span_m=float(np.ptp(PIPER_BODY_OUTLINE[:,1])+2*margin),
                        candidate=list(candidate), applied=list(applied),
                        simulator_geometry_used=False)
            except (KeyError, ValueError, TransformException):
                self.guarded_view_command(0., 0.)
            time.sleep(.05)
        self.guarded_view_command(0., 0.)
        return False

    def reobserve_selected_view(self):
        """Improve a partial view without relaxing the confirmed-handle gate."""
        candidate, basis = (0., 0.), 'no_fresh_robot_transform'
        try:
            transform = self.tf.lookup_transform('base_link', self.reference_frame, Time())
            stamp = transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
            if not 0. <= self.now()-stamp <= .15:
                raise ValueError('Stale current robot transform')
            hint = self.view_hint
            fresh = hint is not None and 0. <= self.now()-hint[0] <= .8
            point = copy.deepcopy(self.request_target)
            if fresh:
                point.point.x, point.point.y, point.point.z = map(float, hint[1])
            current = tf2_geometry_msgs.do_transform_point(point, transform).point
            camera = self.tf.lookup_transform('base_link', 'camera_color_optical_frame', Time()).transform
            q, c = camera.rotation, camera.translation
            optical_yaw = math.atan2(2*(q.y*q.z-q.w*q.x), 2*(q.x*q.z+q.w*q.y))
            candidate = panel_view_command([current.x, current.y], [c.x, c.y], optical_yaw)
            basis = 'fresh_registered_depth_view_hint' if fresh else 'remembered_target_bearing_only'
            if not fresh and candidate[0] != 0.:
                candidate = (0., 0.)
        except (ValueError, TransformException):
            candidate = (0., 0.)
        return candidate, self.guarded_view_command(*candidate), basis

    def panel_observation(self, msg):
        observation = json.loads(msg.data)
        if observation.get('valid') and 'normal' in observation:
            self.panel = observation

    def camera_info(self, msg):
        if msg.k[0] > 0. and msg.width > 0:
            left = math.atan2(msg.k[2], msg.k[0])
            right = math.atan2(msg.width-msg.k[2], msg.k[0])
            self.camera_bearing_limit = max(0., min(left,right)-.04)

    def handle_geometry(self, msg):
        if not self.active or self.request_target is None:
            return
        data=json.loads(msg.data)
        if not 0. <= self.now()-data['stamp'] < .8:
            return
        point=PointStamped()
        point.header.frame_id=data['frame']
        point.header.stamp=Time(seconds=data['stamp']).to_msg()
        point.point.x,point.point.y,point.point.z=map(float,data['point'])
        try:
            mapped=self.transform(point,self.reference_frame).point
        except TransformException:
            return
        if math.hypot(mapped.x-self.request_target.point.x,
                      mapped.y-self.request_target.point.y) <= .90 and data['span_m'] > 0.:
            self.handle_spans.append(data['span_m'])

    def park_from_observation(self):
        """Dock parallel to the measured panel before arm-axis alignment."""
        if not self.reobserve_docking_support():
            return False
        if (self.latest is None or self.panel is None or not self.handle_spans
                or self.camera_bearing_limit is None):
            return False
        if self.now()-self.panel['depth_stamp'] > .6:
            return False
        observed = self.latest[0]
        camera = self.tf.lookup_transform('base_link', 'camera_color_optical_frame', Time()).transform
        self.camera_xyz=np.array([camera.translation.x,camera.translation.y,camera.translation.z])
        anchor = self.transform(observed.handle_position, self.reference_frame).point
        normal = np.asarray(self.panel['normal'][:2], float)
        normal /= np.linalg.norm(normal)
        robot = self.tf.lookup_transform(self.reference_frame, 'base_link', Time()).transform
        q = robot.rotation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if np.dot(normal, [math.cos(yaw), math.sin(yaw)]) < 0:
            normal = -normal
        heading = math.atan2(normal[1], normal[0])
        lateral = np.array([-normal[1], normal[0]])
        along = np.dot(np.array([anchor.x-robot.translation.x, anchor.y-robot.translation.y]), normal)
        minimum=(float(PIPER_BODY_OUTLINE[:,0].max())+self.kinematics.tool_front_extent_m+.015)
        span=float(np.median(self.handle_spans))
        lanes=None
        if self.panel.get('reference_frame')==self.reference_frame and 'segment_xy' in self.panel:
            try:
                lanes=observed_panel_docking_lanes([anchor.x,anchor.y],normal,
                    self.panel['segment_xy'],PIPER_BODY_OUTLINE,
                    self.panel.get('segment_edge_margin_m',.05))
            except ValueError as exc:
                self.event('panel_docking_support_unavailable',reason=str(exc),
                           simulator_geometry_used=False)
                return False
        if lanes is None:
            self.event('panel_docking_support_unavailable',
                       reason='No time-aligned observed panel span',simulator_geometry_used=False)
            return False
        # The chassis must later advance with the held panel. Camera-centred
        # lanes outside the observed panel span cannot substitute for that
        # opening clearance, even if an unloaded grasp is reachable there.
        # Discovery may frame the whole lever; near docking frames its grasp
        # region. Keeping a wide detector ROI in view can exhaust arm reach.
        # The .07 m width is the calibrated two-finger opening, not door truth.
        # Fresh primary YOLO, depth and actual IK still authorize the grasp.
        visibility_span=min(span,.07)
        geometry = dict(camera_xyz=self.camera_xyz, camera_half_fov=self.camera_bearing_limit,
                        candidate_lanes=lanes, prefer_compact=True,
                        pregrasp_clearance_m=self.approach_clearance_floor())
        bounds = [along+.35,self.kinematics.mount_xyz[1],observed.handle_position.point.z]
        posture=self.kinematics.docking_contact_posture(bounds,minimum,
            observed_span_m=visibility_span,**geometry)
        # Rank forward reach in the chosen observed panel-supported lane.
        # These are robot-local reserve probes, never commanded press endpoints.
        posture=self.kinematics.docking_contact_posture(bounds,minimum,
            observed_span_m=visibility_span,**{**geometry,'candidate_lanes':[float(posture[1])]},
            press_reserve_m=.5*.11,press_roll_rad=-.35)
        if span>visibility_span:
            self.event('grasp_view_requires_live_confirmation',observed_span_m=span,
                grasp_view_span_m=visibility_span,primary_confidence_gate=.45)
        self.grip_lane_y=float(posture[1])
        self.docking_posture=posture.copy()
        self.event('camera_workspace_docking_posture', posture=posture.tolist(),
                   observed_handle_span_m=span,grasp_visibility_span_m=visibility_span,
                   panel_supported_lanes=list(map(float,lanes)),
                   lateral_alignment_basis='observed_panel_envelope_then_camera_and_actual_arm')
        self.event('press_workspace_docking_reserve',posture=posture.tolist(),
            predicted_vertical_probe_m=.5*.11,predicted_roll_rad=-.35,
            joint_margin_rad=.20,basis='actual_piper_chain_margin_ranking',
            press_endpoint_prescribed=False,environment_geometry_used=False)
        goal = np.array([anchor.x, anchor.y])-normal*(along+.35)-lateral*self.grip_lane_y
        yaw_error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
        if abs(yaw_error) < .025 and abs(observed.handle_position.point.y-self.grip_lane_y) < .025:
            return True
        self.event('panel_parallel_docking', measured_normal_odom=normal.tolist(), goal_odom=goal.tolist())
        start = self.now()
        report = -1.
        while rclpy.ok() and self.now()-start < 75.:
            # Parking may temporarily turn the camera away from the static
            # observed handle. Live scan/TF guard this bounded manoeuvre;
            # fresh YOLO depth is still mandatory before any arm approach.
            if self.scan is None:
                self.command()
                time.sleep(.05)
                continue
            scan_stamp = self.scan.header.stamp.sec + self.scan.header.stamp.nanosec*1e-9
            transform = self.tf.lookup_transform(self.reference_frame, 'base_link', Time())
            tf_stamp = transform.header.stamp.sec + transform.header.stamp.nanosec*1e-9
            if not (0. <= self.now()-scan_stamp < .75 and 0. <= self.now()-tf_stamp < .75):
                self.command()
                time.sleep(.05)
                continue
            robot = transform.transform
            q = robot.rotation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            dx, dy = goal-np.array([robot.translation.x, robot.translation.y])
            distance = math.hypot(dx, dy)
            yaw_error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
            if distance < .025:
                if abs(yaw_error) < .025:
                    self.command()
                    self.event('panel_parallel_ready', position_error=distance, yaw_error=yaw_error)
                    return True
                self.command(0., np.clip(.8*yaw_error, -.16, .16))
            else:
                alpha = math.atan2(math.sin(math.atan2(dy, dx)-yaw), math.cos(math.atan2(dy, dx)-yaw))
                direction = 1.
                if abs(alpha) > math.pi/2:
                    direction = -1.
                    alpha = math.atan2(math.sin(alpha+math.pi), math.cos(alpha+math.pi))
                beta = math.atan2(math.sin(yaw_error-alpha), math.cos(yaw_error-alpha))
                speed = direction*min(.04, .25*distance)*max(0., math.cos(alpha))
                if not self.safe_motion(speed):
                    self.command()
                else:
                    self.command(speed, np.clip(.8*alpha-.25*beta, -.16, .16))
            if self.now()-report > 5.:
                report = self.now()
                self.event('panel_docking_progress', position_error=distance, yaw_error=yaw_error)
            time.sleep(.05)
        self.command()
        return False

    def safe_forward(self):
        return self.safe_motion(1.)

    def reacquire_camera(self, latest):
        """Back away from an observed static handle before any arm motion."""
        if latest is None or not self.safe_motion(-1.):
            self.command()
            return False
        point = PointStamped()
        point.header.frame_id = self.reference_frame
        point.point.x, point.point.y, point.point.z = map(float, latest[2])
        try:
            current = self.transform(point, 'base_link').point
        except TransformException:
            self.command()
            return False
        if not (.30 < current.x < self.kinematics.contact_reach_upper_bound(current.z)):
            self.command()
            return False
        self.command(-.02, 0.)
        return True

    def alignment_solution(self, xyz):
        if self.docking_posture is None:
            return None, None
        samples=[point for stamp,point in list(self.handle_positions) if 0.<=self.now()-stamp<=.5]
        if len(samples)<3:
            return None,self.docking_posture.copy()
        try:
            _,clearance,_=observed_clearance(samples,self.approach_clearance_floor(),.16)
            bounds=[self.kinematics.contact_reach_upper_bound(xyz[2]),self.grip_lane_y,xyz[2]]
            posture=self.docking_posture.copy()
            if abs(posture[2]-xyz[2])>.003:
                posture=self.kinematics.docking_contact_posture(bounds,
                    float(PIPER_BODY_OUTLINE[:,0].max())+self.kinematics.tool_front_extent_m+.015,
                    camera_xyz=self.camera_xyz,camera_half_fov=self.camera_bearing_limit,
                    observed_span_m=min(float(np.median(self.handle_spans)),.07),
                    candidate_lanes=[self.grip_lane_y],prefer_compact=True,
                    pregrasp_clearance_m=clearance,
                    press_reserve_m=.5*.11,press_roll_rad=-.35)
                self.docking_posture=posture.copy()
            approach=np.asarray(xyz).copy();approach[0]-=clearance
            if self.kinematics.solve(approach,max_iterations=100,position_tolerance_m=.003) is None:
                return None,posture
            solution=self.kinematics.solve(xyz,max_iterations=100,
                minimum_joint_margin_rad=.20)
            if solution is not None:
                pressed=np.asarray(xyz).copy();pressed[2]-=.5*.11
                reserve=self.kinematics.solve(pressed,seed=solution.positions,
                    target_rotation=self.kinematics.approach_rotation@_rot_z(-.35),
                    max_iterations=100,position_tolerance_m=.003,
                    minimum_joint_margin_rad=.20)
                if reserve is None:
                    solution=None
            return solution,posture
        except ValueError:
            return None,self.docking_posture.copy()

    def alignment_forward(self,xyz,posture):
        if posture is None:
            return 0.
        error=float(xyz[0]-posture[0])
        if not math.isfinite(error) or abs(error)<=.002:
            return 0.
        forward=float(np.clip(.6*error,-.02,.04))
        return forward if self.safe_motion(forward) else 0.

    def approach_clearance_floor(self):
        # Same calibrated protrusion, 15 mm tool tolerance and 20 mm approach
        # step used by the physical executor; observation spread is added later.
        return max(0.,self.kinematics.tool_front_extent_m-self.kinematics.tool_contact_offset_m)+.015+.02

    def safe_motion(self, forward):
        if self.scan is None:
            return False
        stamp = self.scan.header.stamp.sec + self.scan.header.stamp.nanosec * 1e-9
        if not 0. <= self.now()-stamp < .75:
            return False
        # A nearby panel is an intentional arm contact target. The chassis
        # must still keep positive clearance from any measured front return.
        centre = 0. if forward >= 0. else math.pi
        values = [r for i, r in enumerate(self.scan.ranges)
                  if abs(math.atan2(math.sin(self.scan.angle_min+i*self.scan.angle_increment-centre),
                                   math.cos(self.scan.angle_min+i*self.scan.angle_increment-centre))) < .35
                  and math.isfinite(r) and r > self.scan.range_min]
        return bool(values) and min(values) > (.16 if forward >= 0. else .55)

    def open(self, request, response):
        if self.active:
            response.message = 'Another contact job is active'
            return response
        self.active = True
        self.sequence += 1
        self.latest = None
        self.view_hint = None
        self.handle_track=ConfirmedHandleTrack(acquire=.45)
        self.panel = None
        self.handle_spans.clear()
        self.handle_positions.clear()
        self.docking_posture = None
        self.command()
        try:
            self.request_target = self.wait_selection_target(request.handle_position)
            self.session.publish(String(data=request.door_id))
            self.event('observe_selected_handle', door_id=request.door_id)
            if not self.park_from_observation():
                response.message = 'Fresh panel-aligned docking not achieved'
                self.event('alignment_failed', reason=response.message)
                return response
            camera = self.tf.lookup_transform('base_link', 'camera_color_optical_frame', Time()).transform
            cam = camera.translation
            q = camera.rotation
            optical_yaw = math.atan2(2*(q.y*q.z-q.w*q.x), 2*(q.x*q.z+q.w*q.y))
            # Keep the camera-centred lane selected from the calibrated TF;
            # the actual PIPER chain validates reach at every new observation.
            grip_lane_y = self.grip_lane_y
            start_wall, start_sim = time.monotonic(), self.now()
            alignment_deadline=SimMotionDeadline(start_sim,start_wall,100.,
                stall_limit=3.,wall_limit=900.)
            stable, used_stamp, ik_future, ik_target = 0, -1., None, None
            report_time = -1.
            reachable = False
            docking_posture = None
            last_ik_stamp = -1.
            camera_reacquired = False
            camera_reacquiring = False
            while rclpy.ok():
                failure=alignment_deadline.failure(self.now(),time.monotonic())
                if failure:
                    self.command()
                    response.message='Selected handle alignment: '+failure
                    self.event('alignment_failed',reason=response.message)
                    return response
                latest = self.latest
                if latest is None or self.now()-latest[1] > .8:
                    if self.now()-report_time > 2.:
                        report_time = self.now()
                        self.event('alignment_wait', reason='selected handle missing or stale')
                    if not camera_reacquiring and latest is not None:
                        self.event('camera_reacquisition_started',
                            basis='lost_fresh_primary_YOLO_before_arm_motion')
                    camera_reacquiring = True
                    if not self.reacquire_camera(latest):
                        self.command()
                    time.sleep(.05)
                    continue
                if camera_reacquiring:
                    self.command()
                    camera_reacquiring = False
                    camera_reacquired = True
                    stable = 0
                    last_ik_stamp = -1.
                observed, stamp, _ = latest
                try:
                    xyz=self.current_handle_xyz(latest)
                except (ValueError,TransformException) as exc:
                    self.command()
                    if self.now()-report_time>2.:
                        report_time=self.now()
                        self.event('alignment_wait',reason=str(exc),original_observation_stamp=stamp)
                    time.sleep(.05)
                    continue
                bearing = math.atan2(xyz[1]-cam.y, xyz[0]-cam.x)-optical_yaw
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                camera_visible = (self.camera_bearing_limit is not None
                                  and abs(bearing) < self.camera_bearing_limit)
                alignment_error = math.atan2(xyz[1]-grip_lane_y, xyz[0])
                if ik_future is not None and ik_future.done():
                    solution, docking_posture = ik_future.result()
                    ratio = (self.kinematics.translation_condition_ratio(solution.positions)
                             if solution is not None else 0.)
                    reachable = (solution is not None and ratio >= .25
                        and np.linalg.norm(xyz-ik_target) < .04
                        and np.all(solution.positions >= PIPER_JOINT_LIMITS[:, 0]+.20-1.e-9)
                        and np.all(solution.positions <= PIPER_JOINT_LIMITS[:, 1]-.20+1.e-9)
                        and docking_posture is not None
                        and xyz[0] >= docking_posture[0]-.015
                        and xyz[0] <= docking_posture[0]+.015)
                    ik_future = None
                if ik_future is None and stamp != last_ik_stamp:
                    last_ik_stamp, ik_target = stamp, xyz.copy()
                    ik_future = self.pool.submit(self.alignment_solution, xyz)
                if abs(alignment_error) < .02 and camera_visible and reachable:
                    self.command()
                    if stamp != used_stamp:
                        stable += 1
                        used_stamp = stamp
                    if stable >= 3:
                        if camera_reacquired:
                            self.event('camera_reacquisition_ready', handle_base=xyz.tolist(),
                                primary_confidence=observed.handle_confidence,
                                joint_margin_rad=.20, minimum_condition_ratio=.25,
                                distinct_observations=stable)
                        self.event('contact_alignment_ready', handle_base=xyz.tolist(), bearing=bearing,
                                   docking_workspace_posture=docking_posture.tolist(),
                                   original_observation_stamp=stamp,
                                   coordinate_basis='exposure_anchored_point_in_current_wheel_odometry_frame')
                        break
                else:
                    stable = 0
                    forward=0.
                    desired=self.docking_posture if docking_posture is None else docking_posture
                    if abs(alignment_error)<.08 and camera_visible and not reachable:
                        if desired is not None:
                            forward=self.alignment_forward(xyz,desired)
                    self.command(forward, np.clip(.6*alignment_error, -.16, .16))
                    if self.now()-report_time > 2.:
                        report_time = self.now()
                        self.event('alignment_progress', handle_base=xyz.tolist(),
                            bearing=bearing, reachable=bool(reachable), forward=forward,
                            alignment_error=alignment_error, observation_age=self.now()-stamp,
                            docking_workspace_posture=(None if desired is None else desired.tolist()))
                time.sleep(.05)
            else:
                response.message = 'No fresh selected YOLO handle with reachable aligned PIPER pose'
                self.event('alignment_failed', reason=response.message)
                return response
            self.command()
            if not self.client.wait_for_service(timeout_sec=5.):
                response.message = 'Physical contact service unavailable'
                return response
            parked = self.wait_precontact_pose()
            observed, exposure, anchored=self.latest
            self.current_handle_xyz(self.latest)
            physical = OpenDoor.Request()
            physical.door_id = request.door_id
            physical.handle_position = copy.deepcopy(observed.handle_position)
            physical.handle_position.header.frame_id=self.reference_frame
            physical.handle_position.point.x,physical.handle_position.point.y,physical.handle_position.point.z=map(float,anchored)
            physical.handle_detected = True
            physical.handle_detection_method = observed.handle_detection_method
            physical.handle_confidence = observed.handle_confidence
            parked_stamp = parked.header.stamp.sec+parked.header.stamp.nanosec*1.e-9
            return_pose = PoseStamped()
            return_pose.header = copy.deepcopy(parked.header)
            return_pose.pose.position.x = parked.transform.translation.x
            return_pose.pose.position.y = parked.transform.translation.y
            return_pose.pose.orientation = copy.deepcopy(parked.transform.rotation)
            self.return_pose_pub.publish(return_pose)
            self.event('precontact_return_pose', frame=return_pose.header.frame_id,
                original_stamp=parked_stamp,
                robot_xy=[return_pose.pose.position.x, return_pose.pose.position.y],
                simulator_geometry_used=False)
            future = self.client.call_async(physical)
            wait_start = time.monotonic()
            while (not future.done() and rclpy.ok()
                   and time.monotonic()-wait_start < CONTACT_OPERATION_WALL_TIMEOUT_SEC):
                time.sleep(.05)
            if not future.done():
                response.message = 'ARM_NOT_STOWED: physical contact operation timeout'
            else:
                result = future.result()
                response.success = result.success
                response.message = result.message if result.success else 'ARM_NOT_STOWED: ' + result.message
            self.event('physical_job_result', door_id=request.door_id,
                       success=response.success, message=response.message)
            return response
        except (TransformException, ValueError) as exc:
            response.message = 'Selected handle localization failed: ' + str(exc)
            self.event('localization_failed', reason=response.message)
            return response
        finally:
            self.command()
            self.session.publish(String(data=''))
            self.active = False
            self.request_target = None

    def wait_precontact_pose(self):
        # TF and /clock arrive independently. Wait stopped for a valid sample;
        # never freshen its timestamp or accept a future/stale return pose.
        deadline = SimMotionDeadline(self.now(), time.monotonic(), 2.,
                                     stall_limit=3., wall_limit=20.)
        last_error = 'no transform received'
        while rclpy.ok():
            self.command()
            failure = deadline.failure(self.now(), time.monotonic())
            if failure:
                break
            try:
                # A later SLAM correction must not move this remembered pose.
                parked = self.tf.lookup_transform('odom', 'base_link', Time())
                stamp = parked.header.stamp.sec+parked.header.stamp.nanosec*1.e-9
                age = self.now()-stamp
                if 0. <= age <= .75:
                    return parked
                last_error = f'measurement age {age:.6f}s'
            except TransformException as exc:
                last_error = str(exc)
            time.sleep(.02)
        raise ValueError('Fresh observed precontact robot pose unavailable: '+last_error)

    def wait_transform(self, point):
        start_wall, start_sim = time.monotonic(), self.now()
        last_error = None
        while rclpy.ok() and self.now()-start_sim <= 2. and time.monotonic()-start_wall <= 10.:
            self.command()
            try:
                return self.transform(point, self.reference_frame)
            except TransformException as exc:
                last_error = exc
                time.sleep(.02)
        raise TransformException('Selected observation TF did not arrive: '+str(last_error))

    def wait_selection_target(self, point):
        # A map-memory landmark selects the door; it is not a new measurement.
        # Fresh RGB-D observations still supply every docking/contact target.
        if point.header.frame_id != 'map':
            return self.wait_transform(point)
        deadline = SimMotionDeadline(self.now(), time.monotonic(), 2.,
                                     stall_limit=3., wall_limit=20.)
        last_error = 'no current map transform'
        while rclpy.ok() and not deadline.failure(self.now(), time.monotonic()):
            self.command()
            try:
                transform = self.tf.lookup_transform(self.reference_frame, 'map', Time())
                stamp = transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
                if not 0. <= self.now()-stamp <= .75:
                    raise ValueError('Current map transform is stale or future')
                target = tf2_geometry_msgs.do_transform_point(point, transform)
                if not all(math.isfinite(v) for v in (target.point.x, target.point.y, target.point.z)):
                    raise ValueError('Nonfinite remembered target')
                target.header.stamp = copy.deepcopy(point.header.stamp)
                self.event('remembered_target_reframed',
                    original_stamp=point.header.stamp.sec+point.header.stamp.nanosec*1.e-9,
                    transform_stamp=stamp, basis='selection_only_not_contact_measurement')
                return target
            except (TransformException, ValueError) as exc:
                last_error = str(exc)
                time.sleep(.02)
        raise TransformException('Selected memory target TF did not arrive: '+last_error)


def main(args=None):
    rclpy.init(args=args)
    node = MissionContactAdapter()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.pool.shutdown(wait=True, cancel_futures=True)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
