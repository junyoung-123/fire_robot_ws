#!/usr/bin/env python3
"""Use the preserved contact controller while saving synchronized raw evidence."""
import importlib.util
import json
from pathlib import Path
import sys

from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('contact_runner', root / 'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

class DemoProbe(runner.ContactProbe):
    def __init__(self, record):
        super().__init__(record)
        self.scan = None
        self.frame_number = 0
        self.last_capture_sim = -1.
        self.demo_file = None
        self.demo_stage = 'OBSERVING'
        self.demo_target_id = None
        self.observation_memory = {}
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

    def scan_cb(self, msg):
        self.scan = msg

    def _detected_door_cb(self, msg):
        super()._detected_door_cb(msg)
        if self.record.yolo_observations:
            item = self.record.yolo_observations[-1]
            if item['door_id'] == msg.door_id and 'odom_xyz' in item:
                self.observation_memory[msg.door_id] = item

    def _image_cb(self, msg):
        super()._image_cb(msg)
        if self._keyframe_dir is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.e-9
        if stamp - self.last_capture_sim < .30:
            return
        self.last_capture_sim = stamp
        folder = self._keyframe_dir.parent / 'demo_frames'
        folder.mkdir(exist_ok=True)
        if self.demo_file is None:
            self.demo_file = (folder.parent / 'demo_timeline.jsonl').open('w', encoding='utf-8')
        images = {}
        for name, source in [('scene', msg), ('robot', self.record.latest_front_image),
                             ('detection', self.record.latest_detection_image),
                             ('overhead', self.record.latest_overhead_image),
                             ('handle', self.record.latest_handle_image)]:
            if source is None:
                continue
            dest = folder / f'{self.frame_number:05d}_{name}.png'
            if runner._save_ros_image(source, dest):
                images[name] = {'file': str(dest.relative_to(folder.parent)),
                                'sim_time': source.header.stamp.sec + source.header.stamp.nanosec * 1.e-9}
        odom = self.record.latest_odom
        pose = None
        if odom:
            p, q = odom.pose.pose.position, odom.pose.pose.orientation
            pose = dict(x=p.x, y=p.y, z=p.z, q=[q.x,q.y,q.z,q.w])
        scan = None
        if self.scan is not None:
            scan = dict(frame=self.scan.header.frame_id, angle_min=self.scan.angle_min,
                        angle_increment=self.scan.angle_increment, range_min=self.scan.range_min,
                        range_max=self.scan.range_max,
                        ranges=[r if runner.math.isfinite(r) else None for r in self.scan.ranges])
        phase = self.record.latest_phase
        row = dict(frame=self.frame_number, sim_time=stamp, wall_elapsed=self._stamp(),
                   stage=self.demo_stage if phase == 'WAITING' else phase,
                   target_id=self.demo_target_id, images=images, pose=pose,
                   observations=list(self.observation_memory.values()), scan=scan,
                   memory_scope='Passive observation cache, not the multi-door mission FSM registry',
                   door_angle_rad=self.record.door_samples[-1][1] if self.record.door_samples else None)
        self.demo_file.write(json.dumps(row, allow_nan=False) + '\n')
        self.demo_file.flush()
        self.frame_number += 1

    def stop_recording(self):
        super().stop_recording()
        if self.demo_file:
            self.demo_file.close()
            self.demo_file = None

original_wait = runner._wait_for_yolo_observation
def wait_for_observation(node, *args, **kwargs):
    node.demo_stage = 'OBSERVING'
    found = original_wait(node, *args, **kwargs)
    if found:
        node.demo_target_id = str(found[0].door_id)
        node.demo_stage = 'TARGET_SELECTED'
    return found

original_approach = runner._approach_visible_handle
def approach(node, args):
    node.demo_stage = 'APPROACH_AND_ALIGN'
    result = original_approach(node, args)
    if result[0] is not None:
        node.demo_stage = 'ALIGNMENT_COMPLETE'
    return result

runner.ContactProbe = DemoProbe
runner._wait_for_yolo_observation = wait_for_observation
runner._approach_visible_handle = approach
if __name__ == '__main__':
    raise SystemExit(runner.main())
