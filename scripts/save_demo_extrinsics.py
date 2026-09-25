#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
rclpy.init()
node=Node('demo_extrinsic_recorder')
buf=Buffer()
listener=TransformListener(buf,node)
result={}
end=time.monotonic()+8
while time.monotonic()<end:
    rclpy.spin_once(node,timeout_sec=.1)
    for frame in ('lidar_link','camera_link','base_footprint'):
        if frame in result:
            continue
        try:
            tf=buf.lookup_transform('base_link',frame,Time())
            p,q=tf.transform.translation,tf.transform.rotation
            result[frame]={'xyz':[p.x,p.y,p.z],'xyzw':[q.x,q.y,q.z,q.w]}
        except TransformException:
            pass
Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
node.destroy_node()
rclpy.shutdown()
