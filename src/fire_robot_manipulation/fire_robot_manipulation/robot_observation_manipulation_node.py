"""Robot encoders and RGB-D only; environmental joint states are forbidden."""
import json
import math
import time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformException
from .minimal_observed_angle_node import MinimalObservedAngle
from .preserved_contact_policy import ManipulationNode
from .piper_actual_kinematics import PIPER_JOINT_NAMES
from .contact_feedback import fresh_sample, SimMotionDeadline, measured_wrist_rotation
from .encoder_articulation import lever_from_wrist, inside_gripper_envelope, gripper_occludes_ray


class RobotObservationManipulation(MinimalObservedAngle):
    def __init__(self):
        self._lever_reference=None
        self._lever_tracking=False
        self._lever_estimate_valid=False
        self._phase='IDLE'
        self._articulation_log_stamp=-1.
        self._door_input_source='rgbd'
        self._depth_confirmation=None
        super().__init__()
        self._contact_anchor_pub=self.create_publisher(PointStamped,'/observed_door/contact_anchor',10)
        self.create_subscription(String,'/observed_door/status',self._panel_status_callback,10,
                                 callback_group=ReentrantCallbackGroup())
        if self._sim_door_feedback_topic!='/observed_door/rotation':
            raise RuntimeError('Environmental joint feedback subscription is forbidden')
        if self._sim_lever_feedback_joint_name!='encoder_lever_rotation':
            raise RuntimeError('Raw lever joint name is forbidden')

    def _set_manip_phase(self,phase,door_id=''):
        self._phase=phase
        if phase=='HANDLE_HELD_BASE_OPEN':
            # Do not feed measured wrist roll back through a growing preload.
            # Keep the achieved latch-clearing posture while following door yaw.
            self._held_press_roll=(self._sim_lever_position-self._sim_lever_initial_position
                +self._feedback_config['press_sign']*self._feedback_config['press_roll_lead_rad'])
            self._feedback_event('latched_press_posture_held',held_roll_rad=self._held_press_roll,
                basis='measured wrist at observed latch release plus existing bounded preload')
        if phase in ('LOCALIZE_HANDLE','PRE_GRASP','RELEASE_HANDLE','RETRACT_FROM_HANDLE','RETURN_HOME'):
            self._lever_tracking=False
            self._lever_estimate_valid=False
        return super()._set_manip_phase(phase,door_id)

    def _feedback_contact_rotation(self):
        if self._phase=='HANDLE_HELD_BASE_OPEN':
            door0,yaw0,rotation=self._feedback_contact_orientation_reference
            return measured_wrist_rotation(rotation,self._sim_door_position-door0,
                self._feedback_base_yaw()-yaw0,self._held_press_roll)
        return super()._feedback_contact_rotation()

    def _sim_door_state_callback(self,msg):
        # Even accidental raw-state injection cannot update either control value.
        if msg.name!=['camera_door_rotation']:
            return
        if not msg.position or not math.isfinite(msg.position[0]): return
        return super()._sim_door_state_callback(msg)

    def _panel_status_callback(self,msg):
        data=json.loads(msg.data)
        if not data.get('valid'): return
        source=data.get('sensor','rgbd')
        if source=='rgbd':
            self._depth_confirmation=(data['depth_stamp'],time.monotonic(),data['angle_rad'])
        self._door_input_source=source

    def _sim_arm_state_callback(self,msg):
        previous=self._feedback_joint_msg_stamp
        super()._sim_arm_state_callback(msg)
        if not self._lever_tracking or previous==self._feedback_joint_msg_stamp:
            return
        try:
            joints=[self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES]
            tool,rotation=self._piper_kinematics.forward(joints)
            transform=self._tf_buffer.lookup_transform('odom','base_link',Time())
            pose_time=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
            joint_time=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
            if abs(pose_time-joint_time)>.10:
                raise ValueError('base_pose_not_aligned_with_joint_measurement')
            q=transform.transform.rotation
            yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            reference,door0,yaw0=self._lever_reference
            angle,residual=lever_from_wrist(reference,rotation,
                self._sim_door_position-door0,float(yaw)-yaw0)
            fingers=self._sim_arm_joint_positions
            opposed=all(.002<v<=.036 for v in (
                fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.)))
            if not opposed:
                raise ValueError('finger_encoders_no_longer_support_opposed_grasp')
            source=self._door_input_source
            self._sim_lever_position=angle
            self._feedback_lever_msg_stamp=self._feedback_joint_msg_stamp
            self._feedback_lever_stamp=self._feedback_arm_stamp
            self._feedback_lever_samples.append((self._feedback_lever_stamp,angle))
            del self._feedback_lever_samples[:-1000]
            self._lever_estimate_valid=True
            self._lever_fit_residual=residual
            if joint_time-self._articulation_log_stamp>=.1:
                self._articulation_log_stamp=joint_time
                self._feedback_event('articulation_estimate',measurement_stamp=joint_time,
                    door_angle_rad=self._sim_door_position,lever_angle_rad=angle,
                    residual_rad=residual,source=source,rigid_grasp_assumed=True)
                point=PointStamped(); point.header=msg.header; point.header.frame_id=self._manipulation_frame
                point.point.x,point.point.y,point.point.z=map(float,tool)
                self._contact_anchor_pub.publish(point)
        except (KeyError,ValueError,TypeError,TransformException) as exc:
            self._lever_estimate_valid=False
            self._lever_invalid_reason=str(exc)

    def _camera_rotation_fresh(self):
        sample=self._depth_confirmation
        return (sample is not None and -.02<=self._sim_time_sec()-sample[0]<=.25
                and fresh_sample(sample[1],time.monotonic(),3.))

    def _feedback_handle_held_open(self,min_delta):
        if not super()._feedback_handle_held_open(min_delta): return False
        self._publish_sim_cmd_vel(0.,0.)
        began=time.monotonic()
        while rclpy.ok() and time.monotonic()-began<3.:
            if (self._camera_rotation_fresh() and self._lever_estimate_valid and
                    abs(self._depth_confirmation[2]-self._sim_door_initial_position)>=min_delta):
                self._feedback_event('opening_visually_reconfirmed',
                    door_delta_rad=abs(self._depth_confirmation[2]-self._sim_door_initial_position))
                return True
            time.sleep(.025)
        return self._feedback_stop('Full opening requires fresh visual reconfirmation before release')

    def _feedback_press(self,handle_pos):
        try:
            _,rotation=self._piper_kinematics.forward([
                self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
            yaw=self._feedback_base_yaw()
            if not self._feedback_fresh():
                return self._feedback_stop('Cannot initialize observed contact baseline')
            self._lever_reference=(rotation.copy(),self._sim_door_position,yaw)
        except (ValueError,KeyError,TransformException) as exc:
            return self._feedback_stop('Contact baseline: '+str(exc))
        # Zero is a relative encoder baseline at grasp, not an environmental reading.
        self._sim_lever_position=0.
        self._sim_lever_initial_position=0.
        self._feedback_lever_samples=[]
        self._feedback_lever_stamp=self._feedback_arm_stamp
        self._lever_tracking=True
        self._lever_estimate_valid=True
        self._feedback_event('encoder_lever_baseline',
            assumption='rigid grasp; latch release still requires observed door motion',
            reference_rotation=rotation.tolist())
        return super()._feedback_press(handle_pos)

    def _feedback_base_yaw(self):
        # TF and /clock arrive independently. Request a common past/current
        # timestamp instead of treating a just-arrived newer TF as stale.
        requested=Time(seconds=self._sim_time_sec())
        began=time.monotonic(); waiting=False; reason='base TF unavailable'
        while rclpy.ok():
            try:
                transform=self._tf_buffer.lookup_transform('odom',self._manipulation_frame,Time())
                latest=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
                if latest>requested.nanoseconds*1.e-9:
                    transform=self._tf_buffer.lookup_transform('odom',self._manipulation_frame,requested)
                stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
                if not fresh_sample(stamp,self._sim_time_sec(),self._feedback_config['max_age_sec']):
                    raise ValueError('Base orientation TF is stale')
                q=transform.transform.rotation
                if not np.all(np.isfinite([q.x,q.y,q.z,q.w])):
                    raise ValueError('Base orientation is not finite')
                if waiting:
                    self._feedback_event('base_tf_recovered',waited_real_sec=time.monotonic()-began)
                return float(np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
            except (TransformException,ValueError) as exc:
                reason=str(exc)
            self._publish_sim_cmd_vel(0.,0.)
            if not waiting:
                self._feedback_event('base_tf_wait',reason=reason,requested_sim_sec=requested.nanoseconds*1.e-9)
                waiting=True
            if time.monotonic()-began>=.6:
                raise ValueError(reason)
            time.sleep(.01)
        raise ValueError('ROS stopped while waiting for base TF')

    def _pregrasp_observation_consistent(self):
        if self._sim_door_initial_position is not None and abs(
                self._sim_door_position-self._sim_door_initial_position)>.03:
            return False
        observed=getattr(self,'_feedback_contact_observation',None)
        anchor=getattr(self,'_feedback_handle_anchor',None)
        if observed is not None and anchor is not None and fresh_sample(observed[0],time.monotonic(),1.):
            old=anchor.point; current=observed[1].point
            if np.linalg.norm(np.array([old.x-current.x,old.y-current.y,old.z-current.z]))>.04:
                point=self._transform_handle_to_manipulation_frame(observed[1])
                if point is None:
                    return False
                tool,rotation=self._piper_kinematics.forward([
                    self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
                p=point.point
                self_hit=inside_gripper_envelope([p.x,p.y,p.z],tool,rotation,
                        self._piper_kinematics.tool_contact_offset_m,
                        self._piper_kinematics.tool_front_extent_m,
                        self._feedback_config['tool_tolerance_m'])
                try:
                    camera=self._tf_buffer.lookup_transform(
                        self._manipulation_frame,'camera_color_optical_frame',Time()).transform.translation
                    retained=self._transform_handle_to_manipulation_frame(anchor).point
                    occluded=gripper_occludes_ray([camera.x,camera.y,camera.z],
                        [retained.x,retained.y,retained.z],tool,rotation,
                        self._piper_kinematics.tool_contact_offset_m,
                        self._piper_kinematics.tool_front_extent_m,
                        self._feedback_config['tool_tolerance_m'])
                except (TransformException,AttributeError):
                    occluded=False
                if self_hit or occluded:
                    self.get_logger().info('Handle depth is self-occluded; retaining stationary visual anchor',
                                           throttle_duration_sec=2.)
                    return True
                self._feedback_event('handle_displacement_before_contact',
                    anchor_odom=[old.x,old.y,old.z],observed_odom=[current.x,current.y,current.z])
                return False
        return True

    def _feedback_move_tool(self,goal,stage):
        # Original bounded servo, with camera-based pre-contact disturbance guard.
        # The preserved version assumes an environmental lever sensor even here.
        if not self._feedback_command_point(goal,stage): return False
        deadline=SimMotionDeadline(self._sim_time_sec(),time.monotonic(),8.,wall_limit=90.)
        stable_since=None; next_sample=self._sim_time_sec(); previous_sample=None
        while rclpy.ok():
            failure=deadline.failure(self._sim_time_sec(),time.monotonic())
            if failure: return self._feedback_stop('Approach step: '+failure)
            actual=self._feedback_tool()
            if actual is None: return self._feedback_stop('Lost feedback while approaching handle')
            error=float(np.linalg.norm(actual-goal))
            if stage=='encoder_contact_follow' and getattr(self,'_feedback_full_handle_follow',False):
                lever=self._feedback_config['press_sign']*(self._sim_lever_position-self._sim_lever_initial_position)
                if lever<self._sim_lever_press_min_angle:
                    return self._feedback_stop('Lever estimate lost while settling held arm')
                if error>self._feedback_config['tracking_limit_m']:
                    return self._feedback_stop('Held arm exceeded tracking limit while settling')
            joint_error=max(abs(self._sim_arm_joint_positions[name]-target)
                            for name,target in zip(PIPER_JOINT_NAMES,self._sim_last_arm_target))
            _,rotation=self._piper_kinematics.forward([self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
            try: desired=self._feedback_contact_rotation()
            except (TransformException,ValueError) as exc:
                return self._feedback_stop('Lost wrist reference during motion: '+str(exc))
            if desired is None: desired=self._piper_kinematics.approach_rotation
            orientation_error=float(np.linalg.norm(self._piper_kinematics._rotation_error(rotation,desired)))
            if self._sim_time_sec()>=next_sample:
                self._feedback_event('approach_sample',stage=stage,actual_tool_xyz=actual.tolist(),
                    requested_xyz=np.asarray(goal).tolist(),error_m=error,
                    orientation_error_rad=orientation_error,joint_tracking_error_rad=float(joint_error))
                if (previous_sample is not None and np.linalg.norm(actual-previous_sample)<.002
                        and (error>self._feedback_config['tool_tolerance_m'] or orientation_error>.10)):
                    measured=np.array([self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
                    candidate=self._feedback_joint_bias+.5*(self._feedback_ideal_arm_target-measured)
                    if np.max(np.abs(candidate))>self._feedback_config['joint_bias_limit_rad']:
                        return self._feedback_stop('Joint bias correction limit exceeded')
                    self._feedback_joint_bias=candidate
                    if not self._feedback_command_point(goal,stage): return False
                    self._feedback_event('servo_correction',joint_bias_rad=candidate.tolist(),
                        measured_error_m=error,orientation_error_rad=orientation_error)
                previous_sample=actual.copy(); next_sample=self._sim_time_sec()+.5
            if stage=='observed_pre_grasp' and not self._pregrasp_observation_consistent():
                return self._feedback_stop('Observed door or handle moved during pre-grasp')
            if error<=self._feedback_config['tool_tolerance_m'] and orientation_error<=.10:
                stable_since=stable_since or self._sim_time_sec()
                if self._sim_time_sec()-stable_since>=.35:
                    self._feedback_event('approach_reached',stage=stage,actual_tool_xyz=actual.tolist(),
                        requested_xyz=np.asarray(goal).tolist(),error_m=error)
                    return True
            else: stable_since=None
            time.sleep(.025)
        return self._feedback_stop('Approach failed to converge; no blind advance')

    def _feedback_fresh(self):
        began=time.monotonic(); paused=False
        while True:
            now=time.monotonic(); maximum=self._feedback_config['max_age_sec']
            stamp=self._feedback_door_msg_stamp
            measured=None if stamp is None else stamp[0]+stamp[1]*1.e-9
            arm=fresh_sample(self._feedback_arm_stamp,now,maximum)
            door=self._angle_freshness.valid(measured,self._feedback_door_stamp,self._sim_time_sec(),now)
            lever=(not self._lever_tracking or (self._lever_estimate_valid and
                   fresh_sample(self._feedback_lever_stamp,now,maximum)))
            if arm and door and lever:
                if paused: self._feedback_event('sensor_stream_recovered',waited_real_sec=now-began)
                return True
            self._publish_sim_cmd_vel(0.,0.)
            if not paused:
                self._feedback_event('sensor_stream_wait',arm_fresh=arm,door_fresh=door,
                                     lever_estimate_fresh=lever,phase=self._phase,
                                     angle_age_sec=None if measured is None else self._sim_time_sec()-measured,
                                     lever_invalid_reason=getattr(self,'_lever_invalid_reason',None))
                paused=True
            if now-began>=.6 or not rclpy.ok(): return False
            time.sleep(.01)

    def _feedback_event(self,event,**values):
        payload=dict(event=event,sim_time_sec=self._sim_time_sec(),
            feedback_source='door:RGB-D plus observed LiDAR panel; lever:measured wrist FK under grasp assumption; arm:robot encoders',
            environmental_joint_feedback_used=False,
            motion_policy='preserved_corridor_success_with_robot_sensor_adapter',**values)
        msg=String(); msg.data=json.dumps(payload,allow_nan=False)
        self._feedback_pub.publish(msg)
        self.get_logger().info('CONTACT_FEEDBACK '+msg.data)


def main(args=None):
    rclpy.init(args=args); node=RobotObservationManipulation()
    executor=MultiThreadedExecutor(num_threads=3); executor.add_node(node)
    try: executor.spin()
    finally:
        executor.shutdown(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
