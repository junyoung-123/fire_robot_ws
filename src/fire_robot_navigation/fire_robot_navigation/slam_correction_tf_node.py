"""Publish a held SLAM correction with live wheel TF on older Humble versions."""
import math

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def correction(map_pose, odom_pose):
    angle = math.atan2(math.sin(map_pose[2]-odom_pose[2]), math.cos(map_pose[2]-odom_pose[2]))
    c, s = math.cos(angle), math.sin(angle)
    return (map_pose[0]-c*odom_pose[0]+s*odom_pose[1],
            map_pose[1]-s*odom_pose[0]-c*odom_pose[1], angle)


class SlamCorrectionTF(Node):
    def __init__(self):
        super().__init__('slam_correction_tf')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.pending = None
        self.current = None
        self.reference = None
        self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.create_timer(.05, self.tick)

    def pose_callback(self, msg):
        if msg.header.frame_id == 'map':
            self.pending = msg

    def tick(self):
        now = self.get_clock().now()
        if self.pending is not None:
            msg = self.pending
            try:
                tf = self.buffer.lookup_transform('odom', 'base_link', Time.from_msg(msg.header.stamp))
            except TransformException:
                tf = None
            if tf is not None:
                p, q = tf.transform.translation, tf.transform.rotation
                mp = msg.pose.pose
                wheel = (p.x, p.y, yaw(q))
                values = (mp.position.x, mp.position.y, yaw(mp.orientation))
                if all(math.isfinite(v) for v in wheel+values):
                    self.current = correction(values, wheel)
                    self.reference = (Time.from_msg(msg.header.stamp), wheel)
                self.pending = None
        if self.current is None:
            return
        try:
            latest = self.buffer.lookup_transform('odom', 'base_link', Time())
        except TransformException:
            return
        if (now-Time.from_msg(latest.header.stamp)).nanoseconds/1e9 > .3:
            return
        p, q = latest.transform.translation, latest.transform.rotation
        stamp, wheel = self.reference
        moved = math.hypot(p.x-wheel[0], p.y-wheel[1]) > .1
        turned = abs(math.atan2(math.sin(yaw(q)-wheel[2]), math.cos(yaw(q)-wheel[2]))) > .1
        if (now-stamp).nanoseconds/1e9 > 2. and (moved or turned):
            return
        x, y, angle = self.current
        out = TransformStamped()
        out.header.stamp = (now+Duration(seconds=.1)).to_msg()
        out.header.frame_id = 'map'
        out.child_frame_id = 'odom'
        out.transform.translation.x, out.transform.translation.y = x, y
        out.transform.rotation.z = math.sin(angle/2)
        out.transform.rotation.w = math.cos(angle/2)
        self.broadcaster.sendTransform(out)


def main(args=None):
    rclpy.init(args=args)
    node = SlamCorrectionTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
