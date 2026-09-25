"""Experimental adapter: replace only door-angle feedback, never fall back."""
import json
import math
import time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode
from fire_robot_manipulation.contact_probe_selector import select_contact_probe
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES, PIPER_JOINT_LIMITS
from fire_robot_manipulation.confirmed_handle_track import ConfirmedHandleTrack
from fire_robot_manipulation.contact_feedback import released_grip_clearance
from fire_robot_manipulation.observed_freshness import ObservedFreshness
from fire_robot_manipulation.contact_feedback import fresh_sample
from fire_robot_manipulation.released_retreat import plan_released_retreat


class ObservedAngleManipulation(ManipulationNode):
    def __init__(self):
        super().__init__()
        self._confirmed_handle_track=ConfirmedHandleTrack(
            acquire=self._feedback_config['handle_min_confidence'])
        self._angle_freshness=ObservedFreshness(bool(self.get_parameter('use_sim_time').value))
        if self._sim_door_feedback_joint_name != 'camera_door_rotation':
            raise RuntimeError('Observed-angle controller must not consume Gazebo door_hinge')
        self.create_subscription(JointState, '/observed_door/rotation',
                                 self._sim_door_state_callback, 10,
                                 callback_group=ReentrantCallbackGroup())

    def _feedback_handle_is_confident(self, msg):
        try:
            measured=self._tf_buffer.transform(msg.handle_position,'odom',timeout=Duration(seconds=.1))
        except TransformException:
            return False
        stamp=msg.handle_position.header.stamp
        return self._confirmed_handle_track.update(
            [measured.point.x,measured.point.y,measured.point.z],
            msg.handle_confidence,stamp.sec+stamp.nanosec*1.e-9)

    def _feedback_event(self, event, **values):
        payload = dict(event=event, sim_time_sec=self._sim_time_sec(),
            feedback_source='door:registered_depth_plane; arm/lever:simulated_encoders', **values)
        msg = String()
        msg.data = json.dumps(payload, allow_nan=False)
        self._feedback_pub.publish(msg)
        self.get_logger().info('CONTACT_FEEDBACK '+msg.data)

    def _feedback_stop(self, reason):
        now = time.monotonic()
        age=lambda stamp: None if stamp is None else now-stamp
        self._feedback_event('source_ages_at_stop', reason=reason,
            camera_angle_age_sec=age(self._feedback_door_stamp),
            arm_encoder_age_sec=age(self._feedback_arm_stamp),
            lever_encoder_age_sec=age(self._feedback_lever_stamp))
        return super()._feedback_stop(reason)

    def _feedback_fresh(self):
        now=time.monotonic()
        maximum=self._feedback_config['max_age_sec']
        stamp=self._feedback_door_msg_stamp
        measured=None if stamp is None else stamp[0]+stamp[1]*1.e-9
        return (fresh_sample(self._feedback_arm_stamp,now,maximum)
                and fresh_sample(self._feedback_lever_stamp,now,maximum)
                and self._angle_freshness.valid(measured,self._feedback_door_stamp,
                                                self._sim_time_sec(),now))

    def _feedback_base_yaw(self):
        # Clock and TF arrive on separate DDS topics. Retry while stopped;
        # never use a future/stale transform to issue a motion command.
        deadline=time.monotonic()+.5
        while True:
            try:
                return super()._feedback_base_yaw()
            except (TransformException,ValueError):
                self._publish_sim_cmd_vel(0.,0.)
                if time.monotonic()>=deadline:
                    raise
                time.sleep(.01)

    def _feedback_handle_held_open(self, min_delta):
        try:
            return super()._feedback_handle_held_open(min_delta)
        except (TransformException,ValueError) as exc:
            return self._feedback_stop('Handle-held feedback unavailable: '+str(exc))

    def _select_contact_probe(self, speed, angular_z, opening_speed):
        if opening_speed is None:
            return speed,angular_z
        if not self._feedback_fresh():
            raise ValueError('Contact planning needs fresh robot feedback')
        door0,yaw0,_=self._feedback_contact_orientation_reference
        angle=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
        normal=np.array([math.cos(angle),math.sin(angle),0.])
        preferred=getattr(self,'_feedback_preferred_contact_posture',self._feedback_encoder_follow_origin)
        speed,angular_z,diagnostic=select_contact_probe(
            self._piper_kinematics,self._feedback_last_cartesian_target.copy(),normal,
            self._feedback_contact_rotation(),self._feedback_ideal_arm_target.copy(),
            self._feedback_joint_bias,preferred,speed,angular_z,opening_speed,
            self._feedback_config['follow_step_m'])
        self._feedback_event('observed_kinematic_probe_selected',linear_x=speed,
                             angular_z=angular_z,**diagnostic)
        return speed,angular_z

    def _feedback_retract_released_handle(self, grip_span):
        self._publish_sim_cmd_vel(0.,0.)
        fingers=self._sim_arm_joint_positions
        half=self._sim_gripper_release_half_width-.002
        if fingers.get('gripper_left_joint',0.)<half or -fingers.get('gripper_right_joint',0.)<half:
            return self._feedback_stop('Released retreat requires both fingers open')
        if self._feedback_tool() is None:
            return self._feedback_stop('Released retreat needs fresh measurements')
        joints=np.array([fingers[n] for n in PIPER_JOINT_NAMES])
        clearance=released_grip_clearance(self._piper_kinematics.tool_front_extent_m,
            self._piper_kinematics.tool_contact_offset_m,grip_span,
            self._feedback_config['tool_tolerance_m'])
        try:
            origin,normal,path=plan_released_retreat(self._piper_kinematics,joints,clearance)
        except ValueError as exc:
            return self._feedback_stop(str(exc))
        # The lever preload has ended. Re-estimate any unloaded servo bias
        # from encoder tracking rather than retaining the loaded compensation.
        self._feedback_joint_bias=np.zeros(6)
        self._feedback_event('released_wrist_retreat_planned',steps=len(path),
            clearance_m=clearance,base_motion=False)
        for point,rotation,_ in path:
            self._feedback_released_rotation=rotation.copy()
            if not self._feedback_move_tool(point,'released_wrist_retreat'):
                return False
        final=self._feedback_tool()
        if final is None:
            return self._feedback_stop('Released retreat verification lost feedback')
        measured=float((origin-final)@normal)
        if measured<clearance-self._feedback_config['tool_tolerance_m']:
            return self._feedback_stop('Released retreat did not clear the measured grip')
        self._feedback_event('released_handle_clearance_verified',
            measured_grip_span_m=grip_span,requested_clearance_m=clearance,
            measured_clearance_m=measured,retraction_mode='unloaded_wrist_arm_retreat')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = ObservedAngleManipulation()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
