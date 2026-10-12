#!/usr/bin/env python3
"""Passive synchronized movie: Gazebo scene, robot bbox camera and live SLAM."""
import argparse
import json
import math
from pathlib import Path
import signal
import time
import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from fire_robot_interfaces.msg import DoorInfo
from ros_gz_interfaces.msg import Contacts
from observed_navigation_recorder import Recorder


class FullRecorder(Recorder):
    def __init__(self, directory):
        super().__init__(directory)
        self.images = {}
        self.grid = None
        self.robot = None
        self.path = []
        self.target = None
        self.memory = []
        self.state = 'STARTING'
        self.phase = 'WAITING'
        self.frame_count = 0
        self.last_frame = -1.
        self.last_crew = 0.
        self.truth_pose = None
        self.reference_angles = {}
        self.reference_stamps = {}
        self.contact_pairs = {}
        self.jobs = []
        self.writer_video = cv2.VideoWriter(str(directory/'full_mission_raw.mp4'),
                                            cv2.VideoWriter_fourcc(*'mp4v'), 15, (1600, 900))
        self.timeline = (directory/'video_timeline.jsonl').open('w', buffering=1)
        (directory/'frames').mkdir(exist_ok=True)
        for name, topic in [('scene', '/proof/perspective/image'),
                            ('raw', '/camera/color/image_raw'),
                            ('bbox', '/door_detection/debug_front')]:
            self.create_subscription(Image, topic, lambda m, n=name: self.image(m, n),
                                     qos_profile_sensor_data)
        self.create_subscription(String, '/manipulation_phase', self.phase_cb, 20)
        self.create_subscription(String, '/contact/mission_events', self.job_cb, 20)
        self.create_subscription(String, '/mission/memory_evidence', self.memory_cb, 10)
        self.create_subscription(String, '/observed_door/status',
                                 lambda m: self.record('angle_observation', payload=json.loads(m.data)), 20)
        self.create_subscription(String, '/manipulation_feedback',
                                 lambda m: self.record('contact_feedback', payload=json.loads(m.data)), 20)
        self.create_subscription(String, '/contact/localization',
                                 lambda m: self.record('contact_localization', payload=json.loads(m.data)), 20)
        self.create_subscription(String,'/observed_door/local_panel',
                                 lambda m: self.record('local_panel_exclusion_observation',payload=json.loads(m.data)),20)
        self.create_subscription(DoorInfo,'/contact/handle_observation',self.handle_observation,20)
        self.create_subscription(LaserScan, '/scan', self.contact_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.wheel_observation, qos_profile_sensor_data)
        for topic in ('/cmd_vel_manual', '/cmd_vel_safe'):
            self.create_subscription(Twist, topic,
                lambda m, t=topic: self.record('chassis_command_observation', topic=t,
                    linear_x=m.linear.x, angular_z=m.angular.z,
                    timestamp_basis='recorder_receive_sim_time'), 20)
        for i in range(1, 4):
            name = f'door_blue{i}'
            self.create_subscription(JointState, f'/evaluation/{name}/joints',
                                     lambda m, n=name: self.reference(m, n), 20)
            for part in ('lever', 'panel', 'latch', 'jamb', 'strike'):
                self.create_subscription(Contacts, f'/proof/audit/{name}/{part}',
                                         lambda m, n=name, p=part: self.contact(m, n, p), 20)
        self.create_timer(5.0, self.checkpoint)

    def handle_observation(self,msg):
        point=msg.handle_position
        self.record('handle_encoder_independent_observation',
            measurement_stamp=point.header.stamp.sec+point.header.stamp.nanosec*1e-9,
            frame=point.header.frame_id,xyz=[point.point.x,point.point.y,point.point.z],
            confidence=msg.handle_confidence,method=msg.handle_detection_method,
            detected=msg.handle_detected)

    def wheel_observation(self, msg):
        if self.phase == 'WAITING' or msg.header.frame_id != 'odom':
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.record('wheel_odometry_observation',
            measurement_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            source='own_wheel_encoders', child_frame=msg.child_frame_id,
            pose=[p.x,p.y,yaw], linear_x=msg.twist.twist.linear.x,
            angular_z=msg.twist.twist.angular.z)

    def contact_scan(self, msg):
        if self.phase == 'WAITING':
            return
        transform = None
        try:
            tf = self.buffer.lookup_transform('odom', msg.header.frame_id, Time.from_msg(msg.header.stamp))
            p, q = tf.transform.translation, tf.transform.rotation
            transform = {'translation': [p.x,p.y,p.z], 'quaternion': [q.x,q.y,q.z,q.w]}
        except Exception:
            pass
        self.record('contact_scan_input', measurement_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                    frame=msg.header.frame_id, angle_min=msg.angle_min, angle_increment=msg.angle_increment,
                    range_min=msg.range_min, range_max=msg.range_max, odom_transform=transform,
                    positive_infinity_indices=[i for i,value in enumerate(msg.ranges)
                                               if math.isinf(value) and value>0.],
                    ranges=[float(value) if math.isfinite(value) else None for value in msg.ranges])

    def map_cb(self, msg):
        self.grid = msg
        super().map_cb(msg)

    def pose_cb(self):
        super().pose_cb()
        try:
            tf = self.buffer.lookup_transform('map', 'base_link', Time())
            p, q = tf.transform.translation, tf.transform.rotation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            self.robot = (p.x, p.y, yaw)
            self.path.append((p.x, p.y))
        except Exception:
            pass

    def truth_cb(self, msg):
        super().truth_cb(msg)
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.truth_pose = (p.x, p.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))

    def state_cb(self, msg):
        super().state_cb(msg)
        self.state = msg.state_description

    def door_cb(self, msg, kind):
        super().door_cb(msg, kind)
        if kind == 'target' and msg.door_color == 'blue':
            p = msg.observed_door_position.point
            self.target = (p.x, p.y, msg.door_id)

    def phase_cb(self, msg):
        self.phase = msg.data
        self.record('physical_phase', phase=msg.data)

    def memory_cb(self, msg):
        payload = json.loads(msg.data)
        self.memory = payload['landmarks']
        self.record('mission_memory', payload=payload)

    def job_cb(self, msg):
        payload = json.loads(msg.data)
        self.record('physical_job', payload=payload)
        if payload['event'] == 'physical_job_result':
            self.jobs.append(payload)

    def reference(self, msg, name):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp-self.reference_stamps.get(name, -1) < .05:
            return
        self.reference_stamps[name] = stamp
        if 'door_hinge' in msg.name:
            value = abs(msg.position[msg.name.index('door_hinge')])
            self.reference_angles[name] = max(self.reference_angles.get(name, 0), value)
            self.record('evaluation_door_angle', model=name, angle_rad=value)
        if 'door_hinge' in msg.name and 'lever_joint' in msg.name:
            self.record('evaluation_articulation', model=name,
                door_angle_rad=msg.position[msg.name.index('door_hinge')],
                lever_angle_rad=msg.position[msg.name.index('lever_joint')])

    def contact(self, msg, name, part):
        if not msg.contacts:
            return
        pairs = [[c.collision1.name, c.collision2.name] for c in msg.contacts]
        key = name+'/'+part
        self.contact_pairs.setdefault(key, set()).update(tuple(p) for p in pairs)
        self.record('evaluation_contact', model=name, part=part, pairs=pairs)

    def checkpoint(self):
        payload = {'video_frames': self.frame_count, 'sim_time': self.seconds(),
                   'state': self.state, 'phase': self.phase,
                   'evaluation_max_angles_rad': self.reference_angles,
                   'physical_jobs': self.jobs,
                   'contact_pairs': {k: sorted(v) for k, v in self.contact_pairs.items()}}
        (self.out/'full_checkpoint.json').write_text(json.dumps(payload, indent=2))

    @staticmethod
    def fit(frame, width, height):
        canvas = np.full((height, width, 3), 235, np.uint8)
        if frame is not None:
            ratio = min(width/frame.shape[1], height/frame.shape[0])
            w, h = int(frame.shape[1]*ratio), int(frame.shape[0]*ratio)
            canvas[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w] = cv2.resize(frame, (w, h))
        return canvas

    def map_image(self, width, height):
        out = np.full((height, width, 3), 245, np.uint8)
        if self.grid is None:
            return out
        g = self.grid
        data = np.asarray(g.data).reshape(g.info.height, g.info.width)
        pixels = np.full((*data.shape, 3), 210, np.uint8)
        pixels[data >= 0] = 250
        pixels[data >= 65] = 45
        pixels = np.flipud(pixels)
        margin = 20
        scale = min((width-2*margin)/g.info.width, (height-2*margin)/g.info.height)
        w, h = round(g.info.width*scale), round(g.info.height*scale)
        left, top = (width-w)//2, (height-h)//2
        out[top:top+h, left:left+w] = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_NEAREST)
        def xy(x, y):
            return (round(left+(x-g.info.origin.position.x)/g.info.resolution*scale),
                    round(top+h-(y-g.info.origin.position.y)/g.info.resolution*scale))
        if len(self.path) > 1:
            cv2.polylines(out, [np.array([xy(*p) for p in self.path], np.int32)], False, (50, 125, 220), 2)
        if self.target is not None:
            cv2.drawMarker(out, xy(*self.target[:2]), (220, 90, 30), cv2.MARKER_CROSS, 18, 3)
        for landmark in self.memory:
            color = max(('blue', 'red', 'green'), key=lambda c: landmark.get(c+'_count', 0))
            bgr = {'blue': (210, 90, 30), 'red': (45, 45, 210), 'green': (50, 170, 50)}[color]
            point = xy(landmark['x'], landmark['y'])
            cv2.circle(out, point, 5, bgr, -1)
            if landmark.get('opened'):
                cv2.drawMarker(out, point, (20, 130, 20), cv2.MARKER_TILTED_CROSS, 14, 2)
        if self.robot:
            x, y, yaw = self.robot
            cv2.arrowedLine(out, xy(x, y), xy(x+.5*math.cos(yaw), y+.5*math.sin(yaw)), (20, 20, 20), 4, tipLength=.4)
        blue = [m for m in self.memory if m.get('blue_count', 0) > max(m.get('red_count', 0), m.get('green_count', 0))]
        cv2.putText(out, f'Observed blue: {len(blue)} | Opened: {sum(bool(m.get("opened")) for m in blue)}',
                    (15, height-10), cv2.FONT_HERSHEY_SIMPLEX, .48, (35, 35, 35), 1)
        return out

    def image(self, msg, name):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self.images[name] = (self.bridge.imgmsg_to_cv2(msg, 'bgr8'), stamp)
        if name != 'scene' or stamp <= self.last_frame:
            return
        self.last_frame = stamp
        camera = self.images.get('bbox')
        camera_kind = 'bbox'
        if camera is None or abs(camera[1]-stamp) >= .2:
            camera = self.images.get('raw')
            camera_kind = 'raw'
        frame = np.full((900, 1600, 3), 255, np.uint8)
        frame[45:855, :920] = self.fit(self.images['scene'][0], 920, 810)
        frame[45:455, 920:] = self.fit(camera[0] if camera and abs(camera[1]-stamp)<1 else None, 680, 410)
        frame[485:855, 920:] = self.map_image(680, 370)
        camera_label = 'ROBOT CAMERA | '+('model detections' if camera_kind == 'bbox' else 'live RGB')
        for text, position in [('GAZEBO | World 1', (20, 30)), (camera_label, (935, 30)),
                               ('LIVE SLAM | observed target + trajectory', (935, 478))]:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, .65, (35, 35, 35), 2)
        stage = self.phase if self.state in ('STARTING', 'OPENING_DOOR') and self.phase != 'WAITING' else self.state
        cv2.putText(frame, f'Sim time {stamp:.1f}s | {stage[:70]}', (20, 883),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (35, 35, 35), 2)
        self.writer_video.write(frame)
        if self.frame_count % 75 == 0:
            cv2.imwrite(str(self.out/'frames'/f'{self.frame_count:06d}.jpg'), frame)
        self.timeline.write(json.dumps({'frame': self.frame_count, 'sim_time': stamp,
            'scene_stamp': stamp, 'camera_stamp': camera[1] if camera else None,
            'camera_kind': camera_kind,
            'state': self.state, 'phase': self.phase})+'\n')
        self.frame_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = FullRecorder(args.output)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while rclpy.ok() and not stopped:
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        node.checkpoint()
        node.writer_video.release()
        node.timeline.close()
        node.csv_file.close()
        node.truth_file.close()
        node.events.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
