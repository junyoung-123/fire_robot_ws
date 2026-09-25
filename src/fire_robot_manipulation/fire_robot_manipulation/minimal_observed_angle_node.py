"""Preserve the successful motion policy; replace only its angle input."""
import json
import time
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from .preserved_contact_policy import ManipulationNode
from .contact_feedback import fresh_sample
from .observed_freshness import ObservedFreshness


class MinimalObservedAngle(ManipulationNode):
    def __init__(self):
        super().__init__()
        if self._sim_door_feedback_joint_name!='camera_door_rotation':
            raise RuntimeError('This adapter requires the observed angle, never door_hinge')
        self._angle_freshness=ObservedFreshness(bool(self.get_parameter('use_sim_time').value))
        self.create_subscription(JointState,'/observed_door/rotation',
            self._sim_door_state_callback,10,callback_group=ReentrantCallbackGroup())

    def _feedback_fresh(self):
        began=time.monotonic()
        paused=False
        while True:
            now=time.monotonic()
            maximum=self._feedback_config['max_age_sec']
            stamp=self._feedback_door_msg_stamp
            measured=None if stamp is None else stamp[0]+stamp[1]*1.e-9
            arm=fresh_sample(self._feedback_arm_stamp,now,maximum)
            lever=fresh_sample(self._feedback_lever_stamp,now,maximum)
            angle=self._angle_freshness.valid(measured,self._feedback_door_stamp,
                                              self._sim_time_sec(),now)
            if arm and lever and angle:
                if paused:
                    self._feedback_event('sensor_stream_recovered', waited_real_sec=now-began)
                return True
            # Keep all original validity gates; wait only with the chassis stopped.
            self._publish_sim_cmd_vel(0.,0.)
            if not paused:
                self._feedback_event('sensor_stream_wait', arm_fresh=arm, lever_fresh=lever,
                                     angle_fresh=angle, measured_stamp=measured,
                                     angle_age_sim_sec=None if measured is None else self._sim_time_sec()-measured)
                paused=True
            if now-began>=.6 or not rclpy.ok():
                return False
            time.sleep(.01)

    def _feedback_event(self,event,**values):
        payload=dict(event=event,sim_time_sec=self._sim_time_sec(),
            feedback_source='door:registered_depth_plane; arm/lever:simulated_encoders',
            motion_policy='preserved_corridor_success',**values)
        msg=String()
        msg.data=json.dumps(payload,allow_nan=False)
        self._feedback_pub.publish(msg)
        self.get_logger().info('CONTACT_FEEDBACK '+msg.data)


def main(args=None):
    rclpy.init(args=args)
    node=MinimalObservedAngle()
    executor=MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
