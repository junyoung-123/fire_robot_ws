#!/usr/bin/env python3
"""Retain the original independent simulator audit alongside camera estimates."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from sensor_msgs.msg import JointState
from std_msgs.msg import String

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('corridor_demo',root/'scripts/run_corridor_demo.py')
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


class ObservedProbe(demo.DemoProbe):
    def __init__(self, record):
        super().__init__(record)
        self.angle_log = None
        self.pending = []
        self.last_reference_stamp = -1.
        self.create_subscription(String, '/observed_door/status', self.angle_cb, 20)
        self.create_subscription(JointState, '/door_joint_states', self.reference_cb, 20)

    def log_angle(self, item):
        item['wall_elapsed'] = self._stamp()
        if self.angle_log is None and self._keyframe_dir is not None:
            self.angle_log = (self._keyframe_dir.parent/'observed_angle_samples.jsonl').open('w')
            for old in self.pending:
                self.angle_log.write(json.dumps(old)+'\n')
            self.pending.clear()
        if self.angle_log is None:
            self.pending.append(item)
        else:
            self.angle_log.write(json.dumps(item)+'\n')
            self.angle_log.flush()

    def angle_cb(self, msg):
        self.log_angle(dict(kind='observation',**json.loads(msg.data)))

    def reference_cb(self, msg):
        if 'door_hinge' in msg.name:
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
            if stamp-self.last_reference_stamp < .019:
                return
            self.last_reference_stamp=stamp
            index = msg.name.index('door_hinge')
            self.log_angle(dict(kind='reference_only',
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9,
                angle_rad=msg.position[index]))

    def stop_recording(self):
        super().stop_recording()
        if self._keyframe_dir is not None:
            meta=dict(world=os.environ.get('CONTACT_FIXTURE_WORLD','physical_contact_door_test.world'),
                      initial_spawn_yaw=float(os.environ.get('CONTACT_INITIAL_YAW','-0.12')),
                      lever_press_sign=float(os.environ.get('CONTACT_LEVER_PRESS_SIGN','-1.0')),
                      angle_controller=os.environ.get('CONTACT_ANGLE_CONTROLLER','observed_angle_manipulation_node'),
                      handle_inference_size=int(os.environ.get('CONTACT_HANDLE_INFERENCE_SIZE','640')),
                      environment_feedback_topic=os.environ.get('CONTACT_ENVIRONMENT_FEEDBACK_TOPIC','/door_joint_states'),
                      lever_feedback_joint=os.environ.get('CONTACT_LEVER_FEEDBACK_JOINT','lever_joint'),
                      scope='Fixture and lever-direction configuration; no door coordinates or hinge angle supplied to controller')
            (self._keyframe_dir.parent/'fixture_configuration.json').write_text(json.dumps(meta,indent=2))
        if self.angle_log is not None:
            self.angle_log.close()
            self.angle_log = None


demo.runner.ContactProbe = ObservedProbe
if __name__ == '__main__':
    raise SystemExit(demo.runner.main())
