#!/usr/bin/env python3
"""Read-only RGB/depth evidence capture for the independent angle experiment."""
from collections import deque
import json
import math
import hashlib
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.time import Time
from tf2_ros import Buffer,TransformListener,TransformException


def stamp(msg):
    return msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9


class Capture(Node):
    def __init__(self, out):
        super().__init__('angle_input_evidence_recorder')
        self.out=out
        out.mkdir(parents=True,exist_ok=True)
        self.depth=deque(maxlen=10)
        self.rgb=deque(maxlen=10)
        self.info=None
        self.bins=set()
        self.bridge=CvBridge()
        self.last_message=time.monotonic()
        self.tf=Buffer()
        self.listener=TransformListener(self.tf,self)
        source=Path(__file__).read_bytes()
        (out/'recorder_source.py').write_bytes(source)
        (out/'recorder_sha256.txt').write_text(hashlib.sha256(source).hexdigest())
        self.create_subscription(Image,'/camera/depth/image_rect_raw',self.depth.append,qos_profile_sensor_data)
        self.create_subscription(Image,'/camera/color/image_raw',self.rgb.append,qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/camera/color/camera_info',self.info_cb,qos_profile_sensor_data)
        self.create_subscription(String,'/observed_door/status',self.angle_cb,10)

    def info_cb(self,msg):
        self.info=msg

    def angle_cb(self,msg):
        self.last_message=time.monotonic()
        observation=json.loads(msg.data)
        if not observation['valid'] or not self.depth or not self.rgb or self.info is None:
            return
        degrees=abs(math.degrees(observation['angle_rad']))
        angle_bin=int(degrees//2)
        if angle_bin in self.bins:
            return
        depth=min(self.depth,key=lambda m:abs(stamp(m)-observation['depth_stamp']))
        rgb=min(self.rgb,key=lambda m:abs(stamp(m)-observation['depth_stamp']))
        if abs(stamp(depth)-observation['depth_stamp'])>.001:
            return
        try:
            transform=self.tf.lookup_transform('odom',rgb.header.frame_id,Time.from_msg(depth.header.stamp)).transform
        except TransformException:
            return
        name=f'{angle_bin:02d}_{degrees:.1f}deg'
        pixels=self.bridge.imgmsg_to_cv2(rgb,'bgr8')
        cv2.imwrite(str(self.out/(name+'_rgb.png')),pixels)
        np.savez_compressed(self.out/(name+'_depth.npz'),depth=self.bridge.imgmsg_to_cv2(depth,'passthrough'))
        meta=dict(observation=observation,depth_stamp=stamp(depth),rgb_stamp=stamp(rgb),
                  optical_frame=rgb.header.frame_id,k=list(self.info.k),
                  width=self.info.width,height=self.info.height,
                  depth_to_odom=dict(translation=[transform.translation.x,transform.translation.y,transform.translation.z],
                                     xyzw=[transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w]),
                  source='Robot camera and registered depth; no hinge state subscription')
        (self.out/(name+'.json')).write_text(json.dumps(meta,indent=2))
        self.bins.add(angle_bin)


def main():
    rclpy.init()
    node=Capture(Path(sys.argv[1]))
    start=time.monotonic()
    try:
        while time.monotonic()-start<1200 and time.monotonic()-node.last_message<45:
            rclpy.spin_once(node,timeout_sec=.2)
    finally:
        print(json.dumps(dict(captured_bins=sorted(node.bins),out=str(node.out))))
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
