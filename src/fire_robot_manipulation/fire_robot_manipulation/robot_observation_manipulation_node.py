"""Robot encoders, RGB-D and LiDAR; environmental joints are forbidden."""
import json
import math
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformException
from .minimal_observed_angle_node import MinimalObservedAngle
from .preserved_contact_policy import ManipulationNode
from .piper_actual_kinematics import PIPER_JOINT_NAMES, PIPER_BODY_OUTLINE, PIPER_STOW
from .contact_feedback import fresh_sample, SimMotionDeadline, measured_wrist_rotation, confirmed_plane_opening, confirmed_angle_progress, released_grip_clearance, grasp_anchored_panel_margin, scanned_body_constraints, ObservedSceneGate, SceneObservationPending, temporal_scan_consensus, measured_preload_release, observed_panel_return_motion, chassis_supplied_opening_speed, contact_scene_planning_margin
from .contact_feedback import observed_panel_barrier_geometry, qualified_panel_support
from .contact_feedback import observed_press_increment
from .contact_feedback import measured_chassis_rest, observed_probe_motion, bounded_probe_pulse_scale
from .contact_feedback import observed_contact_curvature
from .observed_door_plane import rotation_matrix
from .encoder_articulation import lever_from_wrist, articulation_from_wrist, fused_grasped_panel_angle, inside_gripper_envelope, gripper_occludes_ray
from .released_retreat import plan_released_retreat
from .contact_lookahead import plan_contact_motion, continue_contact_plan
from .r34_reference_motion import R34ReferenceMotion
from .confirmed_handle_track import ConfirmedHandleTrack


class NeutralPanelObservationPending(ValueError):
    pass


def neutral_return_task_error(actual,goal,panel_normal_xy,maximum_tangent_shift):
    actual,goal,normal=(np.asarray(value,float) for value in (actual,goal,panel_normal_xy))
    if (actual.shape!=(3,) or goal.shape!=(3,) or normal.shape!=(2,)
            or not all(np.all(np.isfinite(value)) for value in (actual,goal,normal))
            or abs(np.linalg.norm(normal)-1.)>1.e-6
            or not math.isfinite(maximum_tangent_shift) or maximum_tangent_shift<=0.):
        raise ValueError('Invalid measured neutral-return task geometry')
    delta=actual-goal
    tangent=float(delta[:2]@np.array([-normal[1],normal[0]]))
    if abs(tangent)>maximum_tangent_shift:
        raise ValueError('Neutral-return tangent motion exceeds bounded contact step')
    return math.hypot(float(delta[:2]@normal),float(delta[2])),tangent


class RobotObservationManipulation(R34ReferenceMotion, MinimalObservedAngle):
    def __init__(self):
        self._lever_reference=None
        self._lever_tracking=False
        self._lever_estimate_valid=False
        self._lever_alignment_pending=False
        self._lever_observation_state=None
        self._phase='IDLE'
        self._articulation_log_stamp=-1.
        self._door_input_source='rgbd'
        self._depth_confirmation=None
        self._plane_confirmations=deque(maxlen=40)
        self._control_plane_confirmations=deque(maxlen=40)
        self._feedback_measurement_pause_generation=0
        self._contact_scans=deque(maxlen=16)
        self._contact_wheel_samples=deque(maxlen=30)
        self._scene_observation_gate=ObservedSceneGate()
        self._contact_plan_gate=ObservedSceneGate(fail_on_repeated_unsafe=False)
        self._feedback_prepare_contact_stance=True
        self._observed_latch_geometry=[]
        self._encoder_panel_agreement=True
        self._encoder_panel_gate=ObservedSceneGate()
        self._grasp_panel_filter=None
        self._raw_contact_panel=None
        self._arm_observation_history=deque(maxlen=400)
        self._lever_encoder_history=deque(maxlen=20)
        self._observed_panel_segment=None
        self._feedback_observed_contact_curvature=None
        self._confirmed_handle_track=None
        super().__init__()
        self.declare_parameter('contact_motion_policy', 'measured')
        self._contact_motion_policy = self.get_parameter('contact_motion_policy').value
        if self._contact_motion_policy not in ('measured', 'r34_reference'):
            raise ValueError('Unknown contact motion policy')
        # Angle and its quality must be consumed as one observation. Separate
        # DDS topics can arrive in either order, even at the same sensor stamp.
        if self._sim_door_state_subscription is None:
            raise RuntimeError('Missing replaceable observed-angle subscription')
        self.destroy_subscription(self._sim_door_state_subscription)
        self._sim_door_state_subscription=None
        self._contact_anchor_pub=self.create_publisher(PointStamped,'/observed_door/contact_anchor',10)
        self.create_subscription(String,'/observed_door/status',self._panel_status_callback,10,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(LaserScan,'/scan',self._contact_scan_callback,qos_profile_sensor_data,
                                 callback_group=ReentrantCallbackGroup())
        self.create_subscription(Odometry,'/odom',self._contact_wheel_callback,qos_profile_sensor_data,
                                 callback_group=ReentrantCallbackGroup())
        if self._sim_door_feedback_topic!='/observed_door/rotation':
            raise RuntimeError('Environmental joint feedback subscription is forbidden')
        if self._sim_lever_feedback_joint_name!='encoder_lever_rotation':
            raise RuntimeError('Raw lever joint name is forbidden')

    def _feedback_handle_is_confident(self,msg):
        stamp=msg.handle_position.header.stamp
        measured=stamp.sec+stamp.nanosec*1.e-9
        if not fresh_sample(measured,self._sim_time_sec(),.8):
            return False
        try:
            point=self._tf_buffer.transform(msg.handle_position,self._contact_reference_frame,
                                            timeout=Duration(seconds=.1)).point
        except TransformException:
            return False
        xyz=np.array([point.x,point.y,point.z])
        track=self._confirmed_handle_track
        acquire=self._feedback_config['handle_min_confidence']
        if (track is None or (track.anchor is not None and
                np.linalg.norm(xyz-track.anchor)>track.max_shift and
                self._phase in ('IDLE','LOCALIZE_HANDLE','RETURN_HOME') and
                msg.handle_confidence>=acquire)):
            track=self._confirmed_handle_track=ConfirmedHandleTrack(acquire=acquire)
        return track.update(xyz,msg.handle_confidence,measured)

    def _contact_scan_callback(self,msg):
        self._contact_scans.append((msg,time.monotonic()))

    def _contact_wheel_callback(self,msg):
        if msg.header.frame_id!='odom' or msg.child_frame_id not in ('base_link','base_footprint'):
            return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        values=np.array([stamp,p.x,p.y,q.x,q.y,q.z,q.w,
                         msg.twist.twist.linear.x,msg.twist.twist.angular.z])
        if not np.all(np.isfinite(values)) or abs(np.linalg.norm(values[3:7])-1.)>.001:
            return
        yaw=math.atan2(2.*(q.w*q.z+q.x*q.y),1.-2.*(q.y*q.y+q.z*q.z))
        self._contact_wheel_samples.append((stamp,time.monotonic(),np.array([p.x,p.y,yaw])))

    def _feedback_wheel_pose_sample(self):
        now,wall=self._sim_time_sec(),time.monotonic()
        samples=[sample for sample in list(self._contact_wheel_samples)
                 if fresh_sample(sample[0],now,.15) and fresh_sample(sample[1],wall,1.)]
        if not samples:
            raise ValueError('Contact motion needs fresh nonfuture own wheel encoder odometry')
        stamp,_,pose=max(samples,key=lambda sample:sample[0])
        return stamp,pose.copy()

    def _feedback_probe_pose_sample(self):
        # SLAM/scan corrections can change while stopped. They locate the
        # contact, but cannot supply millimetres of chassis motion to the arm.
        generation=getattr(self,'_feedback_measurement_pause_generation',0)
        self._feedback_base_pose_sample()
        sample=self._feedback_wait_wheel_pose_sample()
        if getattr(self,'_feedback_measurement_pause_generation',0)!=generation:
            self._feedback_base_pose_sample()
        return sample

    def _feedback_wait_wheel_pose_sample(self):
        began=None
        while True:
            try:
                sample=self._feedback_wheel_pose_sample()
            except ValueError:
                now,wall=self._sim_time_sec(),time.monotonic()
                self._publish_sim_cmd_vel(0.,0.)
                if not rclpy.ok():
                    raise ValueError('Own wheel pose reobservation interrupted')
                if began is None:
                    began=(now,wall)
                    recent=list(self._contact_wheel_samples)[-3:]
                    self._feedback_event('wheel_pose_reobservation',
                        observation_stamps=[v[0] for v in recent],
                        sim_ages_sec=[now-v[0] for v in recent],
                        receipt_ages_sec=[wall-v[1] for v in recent],
                        reason='no_fresh_nonfuture_own_wheel_pose_while_stopped')
                if now-began[0]>2. or wall-began[1]>30.:
                    raise ValueError('Own wheel pose unavailable after bounded stopped reobservation')
                time.sleep(.025)
                continue
            if began is not None:
                self._feedback_measurement_pause_generation=(
                    getattr(self,'_feedback_measurement_pause_generation',0)+1)
                self._feedback_event('wheel_pose_recovered',measurement_stamp=sample[0],
                    waited_sim_sec=self._sim_time_sec()-began[0],
                    waited_wall_sec=time.monotonic()-began[1],
                    resume_requires_new_sensor_barriers=True)
            return sample

    def _feedback_probe_pulse_scale(self,progress,duration,pulse_duration):
        stamp,_=self._feedback_wait_wheel_pose_sample()
        stamps=sorted(set(s[0] for s in list(self._contact_wheel_samples) if s[0]<=stamp))[-5:]
        if len(stamps)<3:
            raise ValueError('Need independent wheel samples to bound contact pulse latency')
        period=float(max(np.diff(stamps)))
        return bounded_probe_pulse_scale(progress,duration,pulse_duration,
            self._sim_time_sec()-stamp,period)

    def _feedback_contact_ik_reobservation(self,goal,rotation,seed,solution):
        self._publish_sim_cmd_vel(0.,0.)
        now,wall=self._sim_time_sec(),time.monotonic()
        began=getattr(self,'_feedback_contact_ik_pending_since',None)
        if began is None:
            self._feedback_contact_ik_pending_since=began=(now,wall)
        self._feedback_event('contact_ik_reobservation',requested_xyz=np.asarray(goal).tolist(),
            requested_rotation=np.asarray(rotation).tolist(),seed_joints_rad=np.asarray(seed).tolist(),
            candidate_joints_rad=solution.positions.tolist() if solution is not None else None,
            maximum_joint_step_rad=.14,position_tolerance_m=.001,
            independent_fresh_barriers_required=True)
        if now-began[0]>2. or wall-began[1]>30.:
            return False
        self._feedback_measurement_pause_generation=(
            getattr(self,'_feedback_measurement_pause_generation',0)+1)
        return True

    def _feedback_scene_constraints(self):
        now,wall_now=self._sim_time_sec(),time.monotonic()
        candidates=[]
        for msg,received in list(self._contact_scans):
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
            if fresh_sample(stamp,now,.25) and fresh_sample(received,wall_now,2.):
                candidates.append((stamp,msg))
        if not candidates:
            raise ValueError('Contact scene has no current nonfuture LiDAR observation')
        usable=None
        for stamp,msg in sorted(candidates,key=lambda sample: sample[0],reverse=True):
            try:
                transform=self._tf_buffer.lookup_transform_full(
                    self._manipulation_frame,Time(),msg.header.frame_id,
                    Time.from_msg(msg.header.stamp),self._contact_reference_frame)
            except TransformException:
                continue
            usable=stamp,msg,transform
            break
        if usable is None:
            raise ValueError('Contact LiDAR has no time-aligned robot localization transform')
        stamp,msg,transform=usable
        ranges=np.asarray(msg.ranges,float)
        angles=msg.angle_min+np.arange(len(ranges))*msg.angle_increment
        history=[]
        latest_inverse=rotation_matrix([transform.transform.rotation.x,
            transform.transform.rotation.y,transform.transform.rotation.z,
            transform.transform.rotation.w]).T
        latest_translation=np.array([transform.transform.translation.x,
            transform.transform.translation.y,transform.transform.translation.z])
        history_samples=list(self._contact_scans)
        if len(history_samples)<7:
            history_samples=[]
        target_time=Time.from_msg(transform.header.stamp) if history_samples else Time()
        for old,received in history_samples:
            old_stamp=old.header.stamp.sec+old.header.stamp.nanosec*1.e-9
            if (not 0. <= stamp-old_stamp <= .9 or wall_now-received>8.
                    or len(old.ranges)!=len(ranges) or old.angle_increment!=msg.angle_increment
                    or old.angle_min!=msg.angle_min):
                continue
            try:
                tf=self._tf_buffer.lookup_transform_full(self._manipulation_frame,target_time,
                    old.header.frame_id,Time.from_msg(old.header.stamp),self._contact_reference_frame)
            except TransformException:
                continue
            old_ranges=np.asarray(old.ranges,float)
            finite=np.isfinite(old_ranges)&(old_ranges>old.range_min)&(old_ranges<old.range_max)
            xyz=np.column_stack((old_ranges[finite]*np.cos(angles[finite]),
                old_ranges[finite]*np.sin(angles[finite]),np.zeros(int(finite.sum()))))
            q0,t0=tf.transform.rotation,tf.transform.translation
            xyz=(xyz@rotation_matrix([q0.x,q0.y,q0.z,q0.w]).T
                 +np.array([t0.x,t0.y,t0.z])-latest_translation)@latest_inverse.T
            bins=np.rint((np.arctan2(xyz[:,1],xyz[:,0])-msg.angle_min)/msg.angle_increment).astype(int)
            row=np.full(len(ranges),np.inf)
            keep=(bins>=0)&(bins<len(ranges))
            np.minimum.at(row,bins[keep],np.linalg.norm(xyz[keep,:2],axis=1))
            row[~np.isfinite(row)]=np.nan
            history.append((old_stamp,row))
        history=sorted({s:row for s,row in history}.items())
        supported=0
        range_uncertainty=None
        range_reserve=np.zeros(len(ranges))
        raw_ranges=ranges.copy()
        acquired=np.isfinite(raw_ranges)&(raw_ranges>msg.range_min)&(raw_ranges<msg.range_max)
        if len(history)>=7 and history[-1][0]-history[0][0]>=.5:
            ranges,supported,range_uncertainty,range_reserve=temporal_scan_consensus(
                [row for _,row in history],return_uncertainty=True,return_reserve=True)
            ranges[~acquired]=np.nan
        if range_uncertainty is not None:
            self._last_contact_range_uncertainty=(stamp,float(range_uncertainty))
        # range_min is an acquisition limit, not a chassis clearance limit.
        # A positive lower confidence bound below it is still geometry from a
        # valid acquired return. Never clamp that bound upward or discard it.
        uncertain_near=bool(np.any(acquired&np.isfinite(ranges)&(ranges<=0.)))
        if uncertain_near:
            self._publish_sim_cmd_vel(0.,0.)
            indexes=np.flatnonzero(acquired&np.isfinite(ranges)&(ranges<=0.))[:8]
            self._feedback_event('contact_scan_unresolved_lower_bound',scan_stamp=stamp,
                beams=indexes.tolist(),raw_ranges_m=raw_ranges[indexes].tolist(),
                lower_bounds_m=ranges[indexes].tolist())
        valid=acquired&np.isfinite(ranges)&(ranges>0.)&(ranges<msg.range_max)
        points=np.column_stack((ranges[valid]*np.cos(angles[valid]),
                               ranges[valid]*np.sin(angles[valid]),np.zeros(int(valid.sum()))))
        # Compensate motion since scan acquisition using robot localization,
        # not a simulator pose or a static base-to-sensor transform alone.
        q,t=transform.transform.rotation,transform.transform.translation
        rotation=rotation_matrix([q.x,q.y,q.z,q.w])
        translation=np.array([t.x,t.y,t.z])
        points=points@rotation.T+translation
        raw_valid=acquired
        raw_points=np.column_stack((raw_ranges[raw_valid]*np.cos(angles[raw_valid]),
            raw_ranges[raw_valid]*np.sin(angles[raw_valid]),np.zeros(int(raw_valid.sum()))))
        raw_points=raw_points@rotation.T+translation
        raw_gap=(float(np.min(np.linalg.norm(raw_points[:,:2]-np.clip(raw_points[:,:2],
            PIPER_BODY_OUTLINE.min(axis=0),PIPER_BODY_OUTLINE.max(axis=0)),axis=1)))
            if len(raw_points) else 0.)
        if len(points)<3:
            raise ValueError('Contact LiDAR has insufficient finite obstacle observations')
        gaps=np.linalg.norm(points[:,:2]-np.clip(points[:,:2],
            PIPER_BODY_OUTLINE.min(axis=0),PIPER_BODY_OUTLINE.max(axis=0)),axis=1)
        reserve_ranges=ranges[valid]+range_reserve[valid]
        reserve_points=np.column_stack((reserve_ranges*np.cos(angles[valid]),
            reserve_ranges*np.sin(angles[valid]),np.zeros(int(valid.sum()))))
        reserve_points=reserve_points@rotation.T+translation
        reserve_gaps=np.linalg.norm(reserve_points[:,:2]-np.clip(reserve_points[:,:2],
            PIPER_BODY_OUTLINE.min(axis=0),PIPER_BODY_OUTLINE.max(axis=0)),axis=1)
        accounted_clearance=np.maximum(0.,reserve_gaps-gaps)
        if (np.min(gaps)<.15 and
                stamp!=getattr(self,'_last_close_scan_diagnostic_stamp',None)):
            self._last_close_scan_diagnostic_stamp=stamp
            beam_ids=np.flatnonzero(valid)
            nearest=np.argsort(gaps)[:12]
            self._feedback_event('contact_scene_nearest_returns',scan_stamp=stamp,
                robot_time=now,sensor_frame=msg.header.frame_id,
                sensor_to_current_base_translation=[t.x,t.y,t.z],
                sensor_to_current_base_quaternion=[q.x,q.y,q.z,q.w],
                consensus_beams=supported,consensus_scan_stamps=[s for s,_ in history],
                range_uncertainty_margin_m=range_uncertainty,
                raw_body_gap_m=raw_gap,
                returns=[dict(beam=int(beam_ids[k]),range_m=float(ranges[beam_ids[k]]),
                    raw_range_m=float(raw_ranges[beam_ids[k]]) if np.isfinite(raw_ranges[beam_ids[k]]) else None,
                    angle_rad=float(angles[beam_ids[k]]),base_xyz=points[k].tolist(),
                    body_gap_m=float(gaps[k]),
                    neighbor_ranges_m=[float(value) if np.isfinite(value) else None
                        for value in ranges[max(0,beam_ids[k]-3):beam_ids[k]+4]])
                    for k in nearest])
        gate=getattr(self,'_scene_observation_gate',None)
        if gate is None:
            self._scene_observation_gate=gate=ObservedSceneGate()
        gap=float(np.min(gaps))
        if gap<.03 or raw_gap<.03 or uncertain_near or gate.pending is not None:
            self._publish_sim_cmd_vel(0.,0.)
        # Even a modest new close return stops motion immediately. The
        # temporal model cannot hide it while collecting repeated observations.
        gate.observe(stamp,now,wall_now,gap>=.03 and raw_gap>=.03 and not uncertain_near)
        moving=None
        panel=getattr(self,'_observed_panel_segment',None)
        curvature=getattr(self,'_feedback_observed_contact_curvature',None)
        curvature_uncertainty=getattr(self,'_feedback_contact_curvature_uncertainty',None)
        if (getattr(self,'_phase',None) in ('LATCH_CLEAR_PUSH','HANDLE_HELD_BASE_OPEN') and panel is not None
                and curvature is not None and curvature_uncertainty is not None and len(history)>=7
                and fresh_sample(panel[0],now,.25) and fresh_sample(panel[1],wall_now,3.)):
            try:
                data=panel[2]
                tf=self._tf_buffer.lookup_transform(self._manipulation_frame,
                    self._contact_reference_frame,Time())
                q0,t0=tf.transform.rotation,tf.transform.translation
                rotate=rotation_matrix([q0.x,q0.y,q0.z,q0.w])
                shift=np.array([t0.x,t0.y,t0.z])
                segment=np.column_stack((np.array(data['segment_xy']),np.zeros(2)))
                segment=(segment@rotate.T+shift)[:,:2]
                panel_normal=(rotate@np.array(data['normal']))[:2]
                panel_normal/=np.linalg.norm(panel_normal)
                values=np.array([row for _,row in history])
                nominal_ranges=raw_ranges.copy()
                counts=np.isfinite(values).sum(axis=0)
                supported_rays=counts>=7
                nominal_ranges[supported_rays]=np.nanmedian(values[:,supported_rays],axis=0)
                nominal=np.column_stack((nominal_ranges[valid]*np.cos(angles[valid]),
                    nominal_ranges[valid]*np.sin(angles[valid]),np.zeros(int(valid.sum()))))
                nominal=(nominal@rotation.T+translation)[:,:2]
                hand=self._feedback_tool()
                door0,yaw0,_=self._feedback_contact_orientation_reference
                alpha=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
                hand_normal=np.array([math.cos(alpha),math.sin(alpha),0.])
                motion,uncertainty,selected=observed_panel_return_motion(
                    points[:,:2],nominal,segment,panel_normal,hand,hand_normal,curvature,
                    curvature_uncertainty=curvature_uncertainty,
                    edge_margin=data.get('segment_edge_margin_m',.05))
                selected &= supported_rays[valid]
                motion[~selected]=0.; uncertainty[~selected]=0.
                if selected.any():
                    moving=motion,uncertainty
                    if stamp!=getattr(self,'_last_moving_panel_scan_stamp',None):
                        self._last_moving_panel_scan_stamp=stamp
                        self._feedback_event('observed_panel_scene_motion',scan_stamp=stamp,
                            panel_stamp=panel[0],moving_interior_returns=int(selected.sum()),
                            remaining_fixed_returns=int(len(points)-selected.sum()),
                            observed_segment_base_xy=segment.tolist(),curvature_per_m=curvature,
                            segment_source=data.get('segment_source','rgbd'),
                            raw_close_return_stop_unchanged=True)
            except (ValueError,TypeError,TransformException,np.linalg.LinAlgError):
                moving=None
        kwargs={} if moving is None else dict(point_opening_motion=moving[0],motion_uncertainty=moving[1])
        planning_margin = .03
        if getattr(self,'_phase',None)=='HANDLE_HELD_BASE_OPEN':
            planning_margin = contact_scene_planning_margin(
                range_uncertainty, self._feedback_config['tool_tolerance_m'],
                accounted_clearance=accounted_clearance)
            self._feedback_event('contact_scene_planning_reserve',scan_stamp=stamp,
                range_uncertainty_m=range_uncertainty,
                planning_margin_m=float(np.max(planning_margin)),
                planning_margin_min_m=float(np.min(planning_margin)),
                credited_returns=int(np.count_nonzero(accounted_clearance)),
                hard_stop_margin_m=.03,basis='single_count_noise_and_calibrated_pose_tolerance')
        rows,bounds,gap=scanned_body_constraints(points[:,:2],PIPER_BODY_OUTLINE,
                                                margin=planning_margin,**kwargs)
        self._contact_scene_plan_inputs=(points[:,:2].copy(),
            np.zeros(len(points),bool) if moving is None else selected.copy(),
            np.broadcast_to(planning_margin,(len(points),)).copy(),
            np.zeros(len(points)) if moving is None else moving[1].copy())
        self._last_scanned_body_gap=gap
        self._last_scene_scan_stamp=stamp
        return rows,bounds

    def _feedback_scene_command_safe(self,speed,angular,active=False):
        try:
            rows,bounds=self._feedback_scene_constraints()
            if self._last_scanned_body_gap < .03:
                raise ValueError('Observed fixed obstacle violates chassis clearance')
            command=np.array([speed,angular])
            if rows.shape[1]==3:
                hand=self._feedback_tool()
                door0,yaw0,_=self._feedback_contact_orientation_reference
                alpha=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
                normal=np.array([math.cos(alpha),math.sin(alpha),0.])
                command=np.r_[command,chassis_supplied_opening_speed(hand,normal,speed,angular,
                    self._feedback_config['opening_speed_mps'])]
            if not active and np.any(rows@command > bounds+1.e-6):
                self._publish_sim_cmd_vel(0.,0.)
                if self._last_scene_scan_stamp != getattr(self,'_last_scene_replan_stamp',None):
                    self._last_scene_replan_stamp=self._last_scene_scan_stamp
                    self._feedback_event('contact_scene_replan',
                        measured_body_gap_m=self._last_scanned_body_gap,
                        scan_stamp=self._last_scene_scan_stamp,
                        reason='fresh_observation_invalidated_stopped_motion_plan')
                return 'replan'
        except SceneObservationPending as exc:
            self._publish_sim_cmd_vel(0.,0.)
            self._feedback_event('contact_scene_reobservation',reason=str(exc))
            return 'replan'
        except (ValueError,TransformException) as exc:
            return self._feedback_stop('Observed contact scene: '+str(exc))
        return True

    def _feedback_wait_contact_chassis_rest(self):
        if not getattr(self,'_feedback_full_handle_follow',False):
            return True
        self._publish_sim_cmd_vel(0.,0.)
        began,wall = self._sim_time_sec(),time.monotonic()
        samples=[]
        waiting_for_wheels=False
        while rclpy.ok():
            if self._sim_time_sec()-began>3. or time.monotonic()-wall>30.:
                return self._feedback_stop('Chassis did not measurably stop before held probe')
            if not self._feedback_fresh():
                return self._feedback_stop('Lost robot feedback while confirming chassis rest')
            try:
                try:
                    stamp,pose=self._feedback_wheel_pose_sample()
                except ValueError:
                    # DDS and clock callbacks can arrive out of order. Stay
                    # stopped until an original-stamp sample passes the same
                    # freshness checks, within the existing rest deadline.
                    self._publish_sim_cmd_vel(0.,0.)
                    if not waiting_for_wheels:
                        wheel_samples=list(self._contact_wheel_samples)
                        latest=max(wheel_samples,key=lambda value:value[0]) if wheel_samples else None
                        self._feedback_event('chassis_rest_wheel_reobservation',
                            current_sim_sec=self._sim_time_sec(),
                            latest_measurement_stamp=None if latest is None else latest[0],
                            latest_callback_age_sec=None if latest is None else time.monotonic()-latest[1],
                            reason='no_fresh_nonfuture_wheel_sample_while_stopped')
                    waiting_for_wheels=True
                    time.sleep(.025)
                    continue
                waiting_for_wheels=False
                hand=self._feedback_tool()
                if hand is None:
                    raise ValueError('Missing arm feedback during chassis rest')
                if not samples or stamp>samples[-1][0]:
                    samples.append(np.r_[stamp,pose])
                    samples=[sample for sample in samples if stamp-sample[0]<=.25]
                if len(samples)>=3 and measured_chassis_rest(samples[0],samples[-1],hand,
                        .05*self._feedback_config['follow_step_m']):
                    self._feedback_event('chassis_rest_confirmed',
                        first_pose_sample=samples[0].tolist(),last_pose_sample=samples[-1].tolist(),
                        displacement_tolerance_m=.05*self._feedback_config['follow_step_m'],
                        basis='own_wheel_encoder_odometry_without_map_corrections')
                    return True
                if not self._feedback_follow_contact():
                    return False
            except (ValueError,TransformException) as exc:
                return self._feedback_stop('Chassis rest: '+str(exc))
            time.sleep(.025)
        return self._feedback_stop('Chassis rest confirmation interrupted')

    def _feedback_contact_arc_pending(self,cached):
        initial=getattr(self,'_contact_lookahead_arc_pose',None)
        if cached is None or cached[3] is None or initial is None:
            return None
        _,current=self._feedback_probe_pose_sample()
        plan=cached[3]
        self._contact_lookahead_arc_mismatch=False
        v,w=plan.projected_velocity or plan.velocity
        if abs(v)+abs(w)<1.e-9:
            return False
        try:
            _,_,progress=observed_probe_motion(initial,current,cached[1],cached[2],
                v,w,plan.execution_interval_sec,self._feedback_config['follow_step_m'])
        except ValueError:
            # Only the preferred velocity is discarded. Actual short probes
            # retain their original displacement and contact safety limits.
            self._contact_lookahead_arc_mismatch=True
            return False
        return progress<1.

    def _feedback_contact_velocity(self, *args, **kwargs):
        generation=getattr(self,'_feedback_measurement_pause_generation',0)
        gate = getattr(self, '_contact_plan_gate', None)
        if gate is None:
            self._contact_plan_gate = gate = ObservedSceneGate(fail_on_repeated_unsafe=False)
        if (getattr(self,'_phase',None)=='HANDLE_HELD_BASE_OPEN'
                and kwargs.get('observed_curvature') is not None
                and hasattr(self,'_contact_scene_plan_inputs')):
            now=self._sim_time_sec()
            cached=getattr(self,'_contact_lookahead_cache',None)
            point,normal=np.asarray(args[0]),np.asarray(args[1])
            pending=self._feedback_contact_arc_pending(cached)
            if pending is False and not getattr(self,'_contact_lookahead_arc_mismatch',False):
                continuation=continue_contact_plan(cached[3],point,normal,
                    kwargs['observed_curvature'],
                    getattr(self,'_feedback_contact_curvature_uncertainty',None))
                if continuation is not None:
                    self._publish_sim_cmd_vel(0.,0.)
                    _,arc_pose=self._feedback_probe_pose_sample()
                    self._contact_lookahead_arc_pose=arc_pose.copy()
                    cached=(now,point.copy(),normal.copy(),continuation)
                    self._contact_lookahead_cache=cached
                    pending=True
                    self._feedback_event('contact_lookahead_arc_continued',
                        arc_index=continuation.arc_index,arcs=len(continuation.arcs),
                        measured_tool_xyz=point.tolist(),measured_normal_xyz=normal.tolist(),
                        projected_velocity=continuation.projected_velocity,
                        simulator_inputs_used=False,all_live_barriers_required=True)
            if (cached is None or pending is False or (pending is None and (
                    now-cached[0]>8. or np.linalg.norm(point-cached[1])>.015
                    or np.linalg.norm(normal-cached[2])>.04))):
                self._publish_sim_cmd_vel(0.,0.)
                _,arc_pose=self._feedback_probe_pose_sample()
                cloud,moving,reserve,uncertainty=self._contact_scene_plan_inputs
                joints=[self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES]
                options={key:kwargs[key] for key in
                    ('gain','opening_priority_weight','heading_priority_weight',
                     'heading_tolerance_rad','measured_heading_reference') if key in kwargs}
                self._feedback_event('contact_lookahead_input',
                    arm_joints_rad=joints,normal_base_xyz=normal.tolist(),
                    curvature_per_m=kwargs['observed_curvature'],
                    scene_points_base_xy=cloud.tolist(),moving_panel=moving.tolist(),
                    scene_margin_m=reserve.tolist(),motion_uncertainty=uncertainty.tolist(),
                    panel_anchor_base_xyz=np.asarray(kwargs['panel_anchor']).tolist(),
                    panel_normal_base_xyz=np.asarray(kwargs['panel_normal']).tolist(),
                    panel_margin_m=kwargs['margin'],preferred_tool_xyz=np.asarray(args[2]).tolist(),
                    opening_speed_mps=kwargs['opening_speed'],controller_options=options,
                    scan_stamp=self._last_scene_scan_stamp,
                    scope='recorded_robot_inputs_for_offline_replay_not_success_evidence',
                    simulator_inputs_used=False)
                try:
                    plan=plan_contact_motion(self._piper_kinematics,joints,normal,
                        kwargs['observed_curvature'],cloud,moving,reserve,
                        kwargs['panel_anchor'],kwargs['panel_normal'],kwargs['margin'],
                        args[2],opening_speed=kwargs['opening_speed'],
                        moving_uncertainty=uncertainty,
                        controller_options=options)
                except ValueError as exc:
                    self._contact_lookahead_arc_pose=None
                    self._contact_lookahead_cache=(self._sim_time_sec(),point.copy(),normal.copy(),None)
                    self._feedback_event('contact_lookahead_unavailable',reason=str(exc),
                        simulator_inputs_used=False)
                else:
                    self._contact_lookahead_arc_pose=arc_pose.copy()
                    self._contact_lookahead_cache=(self._sim_time_sec(),point.copy(),normal.copy(),plan)
                    self._feedback_event('contact_lookahead_plan',
                        velocity=list(plan.velocity),predicted_opening_rad=plan.predicted_opening_rad,
                        minimum_clearance_m=plan.minimum_clearance_m,
                        minimum_condition=plan.minimum_condition,evaluated=plan.evaluated,
                        projected_velocity=plan.projected_velocity,
                        first_arc_duration_sec=plan.execution_interval_sec,
                        arc_completion_basis='own_wheel_measured_motion_not_elapsed_planning_time',
                        simulator_inputs_used=False,prediction_not_success_evidence=True)
                # The search ran while stopped. Return to the outer loop so
                # panel, encoder and scan barriers are ALL reacquired before
                # any command; none of the pre-search inputs authorize motion.
                raise SceneObservationPending('Stopped while predicting contact arc; reacquire actual barriers')
            if cached is not None and cached[3] is not None:
                # This cached item is only a preferred velocity. Every use
                # still projects it through newly acquired physical barriers.
                plan=cached[3]
                nominal=plan.projected_velocity or plan.velocity
                kwargs=dict(kwargs,nominal_velocity=nominal,
                    nominal_velocity_weight=1.e6 if plan.projected_velocity is not None else 20.,
                    prefer_nominal_recovery=bool(plan.projected_velocity is not None and nominal[0]<0.),
                    # A differential-drive detour can temporarily increase
                    # distance to a soft future posture; physical guards stay.
                    limit_posture_drift=False,allow_posture_recovery=True)
        if getattr(self,'_feedback_measurement_pause_generation',0)!=generation:
            self._publish_sim_cmd_vel(0.,0.)
            raise SceneObservationPending('Wheel reobservation requires new contact velocity barriers')
        def observe_plan(clear):
            now,wall = self._sim_time_sec(),time.monotonic()
            if not fresh_sample(self._last_scene_scan_stamp,now,.25):
                self._publish_sim_cmd_vel(0.,0.)
                began = getattr(self,'_contact_plan_stale_since',None)
                if began is None:
                    self._contact_plan_stale_since = (now,wall)
                elif now-began[0]>2. or wall-began[1]>30.:
                    raise ValueError('Contact velocity plan has no fresh scan after stopped reobservation')
                raise SceneObservationPending('Stopped while replacing stale contact velocity plan')
            self._contact_plan_stale_since = None
            gate.observe(self._last_scene_scan_stamp,now,wall,clear)
        try:
            try:
                command = super()._feedback_contact_velocity(*args, **kwargs)
            except ValueError as exc:
                preparing=(kwargs.get('measured_heading_reference',False)
                    and kwargs.get('heading_tolerance_rad') is None
                    and kwargs.get('limit_posture_drift',False)
                    and kwargs.get('allow_posture_recovery',False)
                    and kwargs.get('arm_constraints') is not None
                    and kwargs.get('scene_constraints') is not None)
                if (str(exc)!='No panel-clear contact motion within calibrated velocity bounds'
                        or not preparing):
                    raise
                # A soft future posture must not forbid a joint-safe escape
                # from a currently observed jamb. All physical barriers stay.
                recovery=dict(kwargs,limit_posture_drift=False)
                command=super()._feedback_contact_velocity(*args,**recovery)
                self._feedback_event('preparing_stance_clearance_recovery',
                    scan_stamp=self._last_scene_scan_stamp,
                    linear_x=command[0],angular_z=command[1],
                    arm_panel_scene_velocity_limits_unchanged=True)
        except ValueError as exc:
            if str(exc) != 'No panel-clear contact motion within calibrated velocity bounds':
                raise
            self._publish_sim_cmd_vel(0., 0.)
            self._feedback_event('contact_plan_reobservation',
                scan_stamp=self._last_scene_scan_stamp,
                body_gap_m=self._last_scanned_body_gap,
                reason='no_safe_velocity_under_current_sensor_constraints')
            if self._last_scene_scan_stamp!=getattr(self,'_last_failed_plan_diagnostic',None):
                self._last_failed_plan_diagnostic=self._last_scene_scan_stamp
                self._feedback_event('contact_velocity_constraints',
                    scan_stamp=self._last_scene_scan_stamp,
                    arguments=[np.asarray(value).tolist() for value in args],
                    constraints={key:([np.asarray(value).tolist() for value in item]
                        if key in ('arm_constraints','scene_constraints') and item is not None
                        else np.asarray(item).tolist() if isinstance(item,np.ndarray) else item)
                        for key,item in kwargs.items()})
            observe_plan(False)
            raise
        if not fresh_sample(self._last_scene_scan_stamp,self._sim_time_sec(),.25):
            observe_plan(True)
        if gate.pending is not None:
            self._publish_sim_cmd_vel(0., 0.)
            observe_plan(True)
        self._contact_plan_stale_since = None
        return command

    def _feedback_latch_motion(self):
        depth=self._feedback_press_depth
        previous_depth=getattr(self,'_latch_roll_depth',None)
        if previous_depth is None or not math.isclose(depth,previous_depth,rel_tol=0.,abs_tol=1.e-9):
            self._latch_roll_depth=depth
            self._latch_press_roll=(self._sim_lever_position-self._sim_lever_initial_position
                +self._feedback_config['press_sign']*self._feedback_config['press_roll_lead_rad'])
        samples=list(self._observed_latch_geometry)
        fit=observed_contact_curvature(samples) if len(samples)>=12 else None
        if fit is None:
            return super()._feedback_latch_motion()
        if not self._feedback_fresh():
            raise ValueError('Latch trajectory requires fresh independent panel and own encoders')
        if not getattr(self,'_feedback_full_handle_follow',False):
            self._publish_sim_cmd_vel(0.,0.)
            if not self._feedback_begin_encoder_follow():
                raise ValueError('Cannot anchor measured coordinated latch motion')
            self._feedback_full_handle_follow=True
            self._feedback_held_height=float(self._feedback_last_cartesian_target[2])
            self._feedback_event('observed_latch_coordinated_motion',**fit,
                basis='observed_panel_headings_and_own_encoder_tcp',
                latch_release_angle_unchanged=True,environment_truth_used=False)
        curvature=fit['curvature_per_m']
        self._feedback_observed_contact_curvature=curvature
        self._feedback_contact_curvature_uncertainty=(
            abs(curvature)*fit['radius_uncertainty_m']/fit['radius_m'])
        actual=self._feedback_tool()
        if actual is None:
            raise ValueError('Observed latch motion lost own arm encoders')
        door0,yaw0,_=self._feedback_contact_orientation_reference
        alpha=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
        normal=np.array([math.cos(alpha),math.sin(alpha),0.])
        anchor,panel_normal,margin=self._feedback_panel_barrier_geometry(actual,normal)
        rows,bounds,_=self._piper_kinematics.contact_velocity_barriers(
            [self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES],
            normal,0.,contact_curvature=curvature)
        advance=self._feedback_config['opening_speed_mps']
        speed,angular,_,_=self._feedback_contact_velocity(
            actual,normal,actual.copy(),PIPER_BODY_OUTLINE,
            margin=margin,panel_anchor=anchor,panel_normal=panel_normal,
            observed_curvature=curvature,door_yaw_rate=0.,opening_speed=advance,
            arm_constraints=(rows,bounds),scene_constraints=self._feedback_scene_constraints())
        return speed,angular,advance

    def _feedback_command_point(self,goal,stage):
        if stage=='feedback_latch_press' and self._phase=='LATCH_CLEAR_PUSH':
            if not self._feedback_fresh():
                return self._feedback_stop('Explicit latch press needs fresh robot feedback')
            # One bounded rotary correction belongs to this explicit press;
            # ordinary following must not integrate the loaded encoder error.
            self._latch_press_roll=(self._sim_lever_position-self._sim_lever_initial_position
                +self._feedback_config['press_sign']*self._feedback_config['press_roll_lead_rad'])
            self._latch_roll_depth=float(self._feedback_press_origin[2]-goal[2])
        return super()._feedback_command_point(goal,stage)

    def _feedback_qualify_probe_reference(self,stamp,goal):
        now=self._sim_time_sec()
        goal=np.asarray(goal,float)
        if (goal.shape!=(3,) or not np.all(np.isfinite(goal))
                or not 0.<=now-stamp<=.15):
            raise ValueError('Contact reference needs a fresh original wheel pose')
        samples=[sample for sample in self._arm_observation_history
                 if 0.<=stamp-sample[0]<=.1]
        if not samples:
            raise ValueError('Contact reference has no aligned original own encoders')
        arm_stamp,joints=samples[-1]
        if now-arm_stamp>.15:
            raise ValueError('Contact reference own encoders are stale at qualification')
        return dict(stamp=stamp,goal=goal.copy(),arm_stamp=arm_stamp,joints=dict(joints))

    def _feedback_commit_probe_points(self,point,measured):
        if hasattr(self,'_feedback_commit_panel_angle'):
            del self._feedback_commit_panel_angle
        samples=list(self._arm_observation_history)
        if not samples:
            raise ValueError('Contact commit has no original-stamp own arm sample')
        target_stamp=point.header.stamp.sec+point.header.stamp.nanosec*1.e-9
        reference=getattr(self,'_feedback_qualified_probe_reference',None)
        qualified_reference=bool(reference is not None
            and abs(reference['stamp']-target_stamp)<=1.e-8
            and np.allclose(reference['goal'],[point.point.x,point.point.y,point.point.z],
                            rtol=0.,atol=1.e-9))
        # Wheel TF and arm callbacks have independent publication phases.
        # Prefer the recent encoder sample already supported at the command
        # time, rather than forcing an unnecessary future-pose extrapolation.
        supported=[sample for sample in samples if 0.<=target_stamp-sample[0]<=.1]
        stamp,joints=supported[-1] if supported else samples[-1]
        if qualified_reference:
            stamp,joints=reference['arm_stamp'],reference['joints']
        xyz,_=self._piper_kinematics.forward([joints[n] for n in PIPER_JOINT_NAMES])
        measured.header.stamp=Time(seconds=stamp).to_msg()
        measured.point.x,measured.point.y,measured.point.z=map(float,xyz)
        began=time.monotonic()
        started=None
        reported=False
        # A dispatched reference was qualified before its IK/pulse. It can
        # remain historical after a stopped sensor wait, but cannot authorize
        # new motion: independent live feedback is still mandatory below.
        qualified=qualified_reference
        while rclpy.ok():
            self._publish_sim_cmd_vel(0.,0.)
            now=self._sim_time_sec()
            if started is None:
                started=now
            if not qualified and any(abs(now-s)>.15 for s in (target_stamp,stamp)):
                raise ValueError('Original contact samples expired while awaiting aligned TF')
            if time.monotonic()-began>30. or now-started>2.:
                raise ValueError('Historical contact transform acquisition deadline exceeded')
            if not self._feedback_fresh():
                raise ValueError('Contact commit lost independent robot feedback')
            if target_stamp>now or stamp>now:
                time.sleep(.005)
                continue
            qualified=True
            try:
                result=super()._feedback_commit_probe_points(point,measured)
                # The transformed points are historical command/measurement
                # references, not fresh motion authorization. Pair their
                # baseline angle at the original encoder time as well.
                history=list(getattr(self,'_control_plane_confirmations',[]))
                paired=[sample for sample in history if abs(sample['stamp']-stamp)<=.1]
                angle=None
                if paired:
                    angle=min(paired,key=lambda sample:abs(sample['stamp']-stamp))['angle']
                else:
                    before=[p for p in history if 0.<=stamp-p['stamp']<=.25]
                    after=[p for p in history if 0.<=p['stamp']-stamp<=.25]
                    if before and after:
                        left=max(before,key=lambda p:p['stamp'])
                        right=min(after,key=lambda p:p['stamp'])
                        span=right['stamp']-left['stamp']
                        change=math.atan2(math.sin(right['angle']-left['angle']),
                                          math.cos(right['angle']-left['angle']))
                        if 0.<span<=.5 and abs(change)<=.10:
                            angle=left['angle']+(stamp-left['stamp'])/span*change
                            self._feedback_event('historical_panel_reference_interpolated',
                                original_encoder_stamp=stamp,
                                supporting_panel_stamps=[left['stamp'],right['stamp']],
                                baseline_angle_rad=angle,fresh_motion_authority=False)
                if getattr(self,'_lever_tracking',False) and angle is None:
                    if not reported:
                        self._feedback_event('contact_commit_panel_reobservation',
                            target_measurement_stamp=target_stamp,arm_measurement_stamp=stamp,
                            reason='Awaiting bracketing original panel observations',
                            measurement_times_preserved=True)
                        reported=True
                    time.sleep(.005)
                    continue
                if angle is not None:
                    self._feedback_commit_panel_angle=angle
                return result
            except TransformException as exc:
                if not reported:
                    self._feedback_event('contact_commit_tf_reobservation',
                        target_measurement_stamp=target_stamp,arm_measurement_stamp=stamp,
                        reason=str(exc),measurement_times_preserved=True)
                    reported=True
                time.sleep(.005)
        raise ValueError('Contact commit interrupted while stopped')

    def _set_manip_phase(self,phase,door_id=''):
        self._phase=phase
        if phase=='LOCALIZE_HANDLE':
            self._feedback_qualified_probe_reference=None
            self._contact_lookahead_cache=None
            self._contact_lookahead_arc_pose=None
            self._contact_plan_gate=ObservedSceneGate(fail_on_repeated_unsafe=False)
            self._contact_plan_stale_since=None
            self._plane_confirmations.clear()
            self._control_plane_confirmations.clear()
            self._observed_latch_geometry=[]
            self._encoder_panel_agreement=True
            self._encoder_panel_gate=ObservedSceneGate()
            self._grasp_panel_filter=None
            self._raw_contact_panel=None
            self._feedback_held_height=None
            self._observed_panel_segment=None
            self._feedback_observed_contact_curvature=None
            self._latch_roll_depth=None
            self._latch_press_roll=None
        if phase=='HANDLE_HELD_BASE_OPEN':
            # Do not feed measured wrist roll back through a growing preload.
            # Keep the achieved latch-clearing posture while following door yaw.
            self._held_press_roll=(self._sim_lever_position-self._sim_lever_initial_position
                +self._feedback_config['press_sign']*self._feedback_config['press_roll_lead_rad'])
            self._feedback_event('latched_press_posture_held',held_roll_rad=self._held_press_roll,
                basis='measured wrist at observed latch release plus existing bounded preload')
            self._feedback_held_height=float(self._feedback_last_cartesian_target[2])
        if phase in ('LOCALIZE_HANDLE','PRE_GRASP','RELEASE_HANDLE','RETRACT_FROM_HANDLE','RETURN_HOME'):
            self._lever_tracking=False
            self._lever_estimate_valid=False
            self._lever_observation_state=None
        return super()._set_manip_phase(phase,door_id)

    def _feedback_contact_rotation(self):
        returning=getattr(self,'_lever_return_rotation',None)
        if returning is not None:
            return returning.copy()
        if self._phase=='LATCH_CLEAR_PUSH' and getattr(self,'_latch_press_roll',None) is not None:
            door0,yaw0,rotation=self._feedback_contact_orientation_reference
            return measured_wrist_rotation(rotation,self._sim_door_position-door0,
                self._feedback_base_yaw()-yaw0,self._latch_press_roll)
        if self._phase=='HANDLE_HELD_BASE_OPEN':
            door0,yaw0,rotation=self._feedback_contact_orientation_reference
            return measured_wrist_rotation(rotation,self._sim_door_position-door0,
                self._feedback_base_yaw()-yaw0,self._held_press_roll)
        return super()._feedback_contact_rotation()

    def _feedback_settle_held_target(self):
        if (self._phase != 'HANDLE_HELD_BASE_OPEN'
                or not getattr(self, '_feedback_full_handle_follow', False)):
            return self._feedback_settle_held_target_once()
        deadline=SimMotionDeadline(self._sim_time_sec(),time.monotonic(),8.,wall_limit=90.)
        # Tangential settling can change measured z under load. Recheck the
        # original unloading reserve before allowing the next chassis probe.
        for _ in range(4):
            failure=deadline.failure(self._sim_time_sec(),time.monotonic())
            if failure:
                return self._feedback_stop('Held height recheck: '+failure)
            if not self._feedback_settle_held_target_once():
                return False
            failure=deadline.failure(self._sim_time_sec(),time.monotonic())
            if failure:
                return self._feedback_stop('Held height recheck: '+failure)
            actual=self._feedback_tool()
            if actual is None or not self._feedback_fresh():
                return self._feedback_stop('Held height recheck needs fresh robot observations')
            goal=self._feedback_last_cartesian_target.copy()
            if (np.shape(actual)!=(3,) or np.shape(goal)!=(3,)
                    or not np.isfinite(actual).all() or not np.isfinite(goal).all()):
                return self._feedback_stop('Held height recheck has invalid measured geometry')
            cfg=self._feedback_config
            door0,yaw0,_=self._feedback_contact_orientation_reference
            heading=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
            normal=np.array([math.cos(heading),math.sin(heading),0.])
            residual=actual-goal
            if (np.linalg.norm(residual)>cfg['tracking_limit_m']
                    or abs(residual@normal)>cfg['tool_tolerance_m']
                    or abs(residual[2])>cfg['tool_tolerance_m']):
                return self._feedback_stop('Held height recheck exceeds original pose bounds')
            if not (.5*cfg['tool_tolerance_m']+1.e-8 < residual[2]
                    and actual[2]<self._measured_prepress_height):
                return True
            self._publish_sim_cmd_vel(0.,0.)
            self._feedback_event('held_height_recheck_before_base',
                actual_tool_xyz=actual.tolist(),requested_tool_xyz=goal.tolist(),
                residual_z_m=float(residual[2]),
                basis='fresh_own_encoders_after_stopped_tangential_settling')
        return self._feedback_stop('Held height reserve did not settle before base motion')

    def _feedback_settle_held_target_once(self):
        if self._phase not in ('LATCH_CLEAR_PUSH', 'HANDLE_HELD_BASE_OPEN'):
            return super()._feedback_settle_held_target()
        self._publish_sim_cmd_vel(0.,0.)
        actual=self._feedback_tool()
        goal=self._feedback_last_cartesian_target.copy()
        if self._feedback_loaded_settle_goal(goal,actual) is None:
            return False
        return super()._feedback_settle_held_target()

    def _feedback_confirmed_hold_reference(self,previous_goal,goal):
        following=(getattr(self,'_feedback_full_handle_follow',False)
                   and getattr(self,'_feedback_follow_mode',None)=='encoder_lever_contact'
                   and self._phase in ('LATCH_CLEAR_PUSH','HANDLE_HELD_BASE_OPEN'))
        if not following:
            return True
        # Retain only the explicit, response-confirmed press command's shift.
        # Rebasing every encoder sample would integrate load error into motion.
        shift=goal-previous_goal
        shift[2]=0.
        actual=self._feedback_tool()
        cfg=self._feedback_config
        if (actual is None or not self._feedback_fresh()
                or not np.isfinite(shift).all()
                or np.linalg.norm(shift)>cfg['tool_tolerance_m']):
            return self._feedback_stop('Confirmed hold correction cannot update contact reference')
        door0,yaw0,_=self._feedback_contact_orientation_reference
        heading=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
        normal=np.array([math.cos(heading),math.sin(heading),0.])
        residual=actual-goal
        if (not np.isfinite(residual).all()
                or np.linalg.norm(residual)>cfg['tracking_limit_m']
                or abs(residual@normal)>cfg['tool_tolerance_m']
                or abs(residual[2])>cfg['tool_tolerance_m']):
            return self._feedback_stop('Confirmed hold correction exceeds original pose bounds')
        try:
            transform=self._tf_buffer.lookup_transform(
                self._contact_reference_frame,self._manipulation_frame,Time(),
                timeout=Duration(seconds=.1))
            q=transform.transform.rotation
            fixed_shift=rotation_matrix([q.x,q.y,q.z,q.w])@shift
        except (ValueError,TransformException) as exc:
            return self._feedback_stop('Confirmed hold reference transform: '+str(exc))
        reference,previous,angle=self._feedback_sweep_state
        self._feedback_sweep_state=(reference+fixed_shift,previous,angle)
        self._feedback_press_origin[:2]=goal[:2]
        self._feedback_event('confirmed_hold_reference_shift',
            previous_target_xyz=previous_goal.tolist(),requested_tool_xyz=goal.tolist(),
            fixed_reference_shift_xyz=fixed_shift.tolist(),
            actual_tool_xyz=actual.tolist(),
            basis='response_confirmed_bounded_press_command_not_passive_encoder_drift')
        return True

    def _feedback_loaded_settle_goal(self,goal,actual):
        if self._phase not in ('LATCH_CLEAR_PUSH', 'HANDLE_HELD_BASE_OPEN'):
            return goal
        cfg=self._feedback_config
        if actual is None or not self._feedback_fresh():
            self._feedback_stop('Preload release needs fresh robot observations')
            return None
        door0,yaw0,_=self._feedback_contact_orientation_reference
        alpha=self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
        normal=np.array([math.cos(alpha),math.sin(alpha),0.])
        residual=actual-goal
        if (
                not np.isfinite(residual).all()
                or np.linalg.norm(residual)>cfg['tracking_limit_m']
                or abs(residual@normal)>cfg['tool_tolerance_m']
                or abs(residual[2])>cfg['tool_tolerance_m']):
            self._feedback_event('loaded_pose_bound_exceeded',
                requested_tool_xyz=np.asarray(goal).tolist(),actual_tool_xyz=actual.tolist(),
                normal_error_m=float(residual@normal),vertical_error_m=float(residual[2]))
            self._feedback_stop('Loaded preload exceeds original pose bounds')
            return None
        # Relieve measured excess preload during latch probes as well, before
        # it can exceed the unchanged 15 mm bound at the opening handoff.
        unloading_supported=(
            .5*cfg['tool_tolerance_m'] < residual[2] <= cfg['tool_tolerance_m']
            and abs(residual@normal)<=cfg['tool_tolerance_m']
            and np.linalg.norm(residual)<=cfg['tracking_limit_m']
            and actual[2] < self._measured_prepress_height)
        if unloading_supported:
            try:
                joints=[self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES]
                _,rotation=self._piper_kinematics.forward(joints)
                desired=self._feedback_contact_rotation()
                if np.linalg.norm(self._piper_kinematics._rotation_error(rotation,desired)) > .10:
                    raise ValueError('Wrist orientation is not retained during preload release')
                fingers=self._sim_arm_joint_positions
                if not all(.002<v<=.036 for v in (
                        fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.))):
                    raise ValueError('Opposed grip is not retained during preload release')
                roll=cfg['press_sign']*(self._sim_lever_position-self._sim_lever_initial_position)
                released=measured_preload_release(goal,actual,normal,
                    cfg['tracking_limit_m'],cfg['follow_step_m'],cfg['tool_tolerance_m'],
                    self._measured_prepress_height,roll,self._sim_lever_press_min_angle)
                depth=float(self._feedback_press_origin[2]-released[2])
                if not 0.<=depth<=cfg['press_max_travel_m']:
                    raise ValueError('Released preload lies outside the original press travel bounds')
                self._publish_sim_cmd_vel(0.,0.)
                if not self._feedback_command_point(released,'encoder_contact_follow'):
                    return None
                # Following rebuilds z from the press origin and depth; retain
                # this measured unloading instead of restoring the old preload.
                previous_depth=self._feedback_press_depth
                self._feedback_press_depth=depth
                self._feedback_held_height=float(released[2])
                self._feedback_event('measured_vertical_preload_released',
                    actual_tool_xyz=actual.tolist(),previous_target_xyz=goal.tolist(),
                    requested_tool_xyz=released.tolist(),retained_roll_rad=roll,
                    previous_commanded_depth_m=previous_depth,retained_commanded_depth_m=depth,
                    normal_shift_m=float((released-goal)@normal),force_sensor_used=False,
                    basis='own_encoder_residual_with_retained_grip_and_wrist_roll')
                goal=released
            except (ValueError,TransformException) as exc:
                self._feedback_stop('Measured preload release: '+str(exc))
                return None
        return goal

    def _sim_door_state_callback(self,msg):
        # Even accidental raw-state injection cannot update either control value.
        if msg.name!=['camera_door_rotation']:
            return
        if not msg.position or not math.isfinite(msg.position[0]): return
        raw_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
        previous=getattr(self,'_grasp_panel_filter',None)
        if previous is not None and raw_stamp<=previous[2]:
            return
        self._raw_contact_panel=(raw_stamp,float(msg.position[0]))
        if getattr(self,'_lever_tracking',False):
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
            try:
                history=list(getattr(self,'_arm_observation_history',[]))
                if history:
                    joint_time,aligned_joints=min(history,key=lambda sample:abs(sample[0]-stamp))
                else:
                    joint_stamp=self._feedback_joint_msg_stamp
                    if joint_stamp is None:
                        raise ValueError('missing_grasp_encoder_stamp')
                    joint_time=joint_stamp[0]+joint_stamp[1]*1.e-9
                    aligned_joints=self._sim_arm_joint_positions.copy()
                if abs(stamp-joint_time)>.1:
                    raise ValueError('panel_and_encoders_not_time_aligned')
                _,rotation=self._piper_kinematics.forward([
                    aligned_joints[n] for n in PIPER_JOINT_NAMES])
                latest_tf=self._tf_buffer.lookup_transform(self._contact_reference_frame,'base_link',Time())
                latest_time=latest_tf.header.stamp.sec+latest_tf.header.stamp.nanosec*1.e-9
                # Encoder publication can lead wheel TF by a few milliseconds.
                # Use available pose data at the panel time, never extrapolate.
                tf=(latest_tf if latest_time<=stamp else self._tf_buffer.lookup_transform(
                    self._contact_reference_frame,'base_link',Time(seconds=stamp)))
                pose_stamp=tf.header.stamp.sec+tf.header.stamp.nanosec*1.e-9
                if max(abs(pose_stamp-joint_time),abs(pose_stamp-stamp))>.1:
                    raise ValueError('grasp_base_orientation_not_time_aligned')
                q=tf.transform.rotation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                reference,door0,yaw0=self._lever_reference
                fingers=self._sim_arm_joint_positions
                if not all(.002<v<=.036 for v in (
                        fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.))):
                    raise ValueError('grasp_encoders_no_longer_support_fusion')
                encoder_yaw,_,_=articulation_from_wrist(reference,rotation,yaw-yaw0)
                encoder_angle=float(door0+encoder_yaw)
                baseline=(float(msg.position[0]),encoder_angle) if previous is None else previous[:2]
                qualities = [sample['uncertainty'] for sample in self._plane_confirmations
                    if abs(sample['stamp']-stamp) <= .15
                    and fresh_sample(sample['stamp'], self._sim_time_sec(), .5)
                    and fresh_sample(sample['received'], time.monotonic(), 3.)
                    and math.isfinite(sample['uncertainty'])
                    and 0. <= sample['uncertainty'] <= math.radians(2.)]
                if not qualities:
                    raise SceneObservationPending('independent_panel_confidence_pending')
                panel_uncertainty = max(qualities)
                angle,disagreement,_,residual=fused_grasped_panel_angle(
                    msg.position[0],reference,rotation,yaw-yaw0,door0,previous_estimate=baseline,
                    panel_uncertainty_rad=panel_uncertainty)
                self._encoder_panel_gate.observe(stamp,self._sim_time_sec(),time.monotonic(),True)
                self._encoder_panel_agreement=True
                self._grasp_panel_filter=(angle,encoder_angle,stamp)
                self._feedback_event('grasped_panel_fusion',panel_angle_rad=float(msg.position[0]),
                    control_angle_rad=angle,panel_encoder_disagreement_rad=disagreement,
                    encoder_residual_rad=float(residual),measurement_stamp=stamp,
                    independent_panel_uncertainty_rad=panel_uncertainty,
                    raw_panel_preserved_for_completion=True,rigid_grasp_assumed=True,
                    prediction_basis='encoder_increment_bounded_by_independent_panel_confidence')
                filtered=JointState()
                filtered.header=msg.header
                filtered.name=msg.name
                filtered.position=[angle]
                msg=filtered
            except (ValueError,AttributeError,KeyError,TransformException,SceneObservationPending) as exc:
                self._encoder_panel_agreement=False
                self._publish_sim_cmd_vel(0.,0.)
                # Missing temporal alignment is acquisition latency, not a
                # contradictory geometric measurement. Motion still stops,
                # but it cannot poison the independent contradiction gate.
                alignment_pending=(isinstance(exc,TransformException) or
                    str(exc) in ('missing_grasp_encoder_stamp','panel_and_encoders_not_time_aligned',
                                 'grasp_base_orientation_not_time_aligned'))
                if not isinstance(exc,SceneObservationPending) and not alignment_pending:
                    try:
                        self._encoder_panel_gate.observe(stamp,self._sim_time_sec(),time.monotonic(),False)
                    except (ValueError,SceneObservationPending):
                        pass
                self._feedback_event('grasped_panel_reobservation',reason=str(exc),measurement_stamp=stamp)
                return
        result=super()._sim_door_state_callback(msg)
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
        self._control_plane_confirmations.append(dict(stamp=stamp,received=time.monotonic(),
                                                     angle=float(msg.position[0])))
        return result

    def _panel_status_callback(self,msg):
        data=json.loads(msg.data)
        if not data.get('valid'): return
        source=data.get('sensor','rgbd')
        if source not in ('rgbd','lidar') or data.get('reference_frame') != self._contact_reference_frame:
            return
        if 'segment_xy' in data and 'normal' in data:
            self._observed_panel_segment=(data['depth_stamp'],time.monotonic(),data)
        if data.get('angle_measurement_valid') is False or 'angle_rad' not in data:
            return
        if source=='rgbd':
            self._depth_confirmation=(data['depth_stamp'],time.monotonic(),data['angle_rad'])
        self._door_input_source=source
        self._plane_confirmations.append(dict(stamp=data['depth_stamp'],received=time.monotonic(),
            angle=data['angle_rad'],uncertainty=data.get('orientation_uncertainty_rad',float('nan')),
            sensor=source, center=data.get('center'), normal=data.get('normal'),
            range_uncertainty=(data.get('plane_offset_uncertainty_m') if source=='lidar' else None),
            residual=data.get('residual_m')))
        measurement=JointState()
        measurement.header.stamp=Time(seconds=float(data['depth_stamp'])).to_msg()
        measurement.header.frame_id=self._contact_reference_frame
        measurement.name=['camera_door_rotation']
        measurement.position=[float(data['angle_rad'])]
        self._sim_door_state_callback(measurement)

    def _sim_arm_state_callback(self,msg):
        previous=self._feedback_joint_msg_stamp
        super()._sim_arm_state_callback(msg)
        history=getattr(self,'_arm_observation_history',None)
        if history is not None and previous!=self._feedback_joint_msg_stamp:
            stamp=self._feedback_joint_msg_stamp
            if stamp is not None:
                measured_stamp=stamp[0]+stamp[1]*1.e-9
                if not history or measured_stamp>history[-1][0]:
                    history.append((measured_stamp,self._sim_arm_joint_positions.copy()))
        if previous!=self._feedback_joint_msg_stamp and self._feedback_joint_msg_stamp is not None:
            self._lever_encoder_history.append((self._feedback_joint_msg_stamp,
                self._feedback_arm_stamp,self._sim_arm_joint_positions.copy()))
        if not self._lever_tracking or previous==self._feedback_joint_msg_stamp:
            return
        # A robot-local FK anchor does not require environmental localization.
        # Keep RGB-D panel exclusion observable even while global TF is pending.
        local_now,local_wall=self._sim_time_sec(),time.monotonic()
        eligible=[sample for sample in list(self._lever_encoder_history)
            if fresh_sample(sample[0][0]+sample[0][1]*1.e-9,local_now,.15)
            and fresh_sample(sample[1],local_wall,self._feedback_config['max_age_sec'])]
        if eligible:
            local_stamp,_,local_positions=max(eligible,key=lambda sample:sample[0])
            local_time=local_stamp[0]+local_stamp[1]*1.e-9
            if local_time>getattr(self,'_feedback_local_anchor_stamp',-1.):
                local_joints=[local_positions[n] for n in PIPER_JOINT_NAMES]
                local_tool,_=self._piper_kinematics.forward(local_joints)
                local_point=PointStamped()
                local_point.header.stamp=Time(seconds=local_time).to_msg()
                local_point.header.frame_id=self._manipulation_frame
                local_point.point.x,local_point.point.y,local_point.point.z=map(float,local_tool)
                self._contact_anchor_pub.publish(local_point)
                self._feedback_local_anchor_stamp=local_time
        try:
            now,wall=self._sim_time_sec(),time.monotonic()
            unavailable=TransformException('No fresh robot encoder sample with matching TF')
            # TF and encoders arrive independently. Select the newest fresh
            # common snapshot, retaining its original measurement/receipt times.
            for joint_stamp,received,fingers in reversed(list(self._lever_encoder_history)):
                joint_time=joint_stamp[0]+joint_stamp[1]*1.e-9
                if (not fresh_sample(joint_time,now,.15) or
                        not fresh_sample(received,wall,self._feedback_config['max_age_sec'])):
                    continue
                try:
                    transform=self._tf_buffer.lookup_transform(
                        self._contact_reference_frame,'base_link',Time(seconds=joint_time))
                except TransformException as exc:
                    unavailable=exc
                    continue
                break
            else:
                raise unavailable
            joints=[fingers[n] for n in PIPER_JOINT_NAMES]
            tool,rotation=self._piper_kinematics.forward(joints)
            pose_time=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
            if abs(pose_time-joint_time)>.10:
                raise ValueError('base_pose_not_aligned_with_joint_measurement')
            q=transform.transform.rotation
            yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            reference,door0,yaw0=self._lever_reference
            angle,residual=lever_from_wrist(reference,rotation,
                self._sim_door_position-door0,float(yaw)-yaw0)
            opposed=all(.002<v<=.036 for v in (
                fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.)))
            if not opposed:
                raise ValueError('finger_encoders_no_longer_support_opposed_grasp')
            source=self._door_input_source
            self._sim_lever_position=angle
            self._feedback_lever_msg_stamp=joint_stamp
            self._feedback_lever_stamp=received
            self._feedback_lever_samples.append((self._feedback_lever_stamp,angle))
            del self._feedback_lever_samples[:-1000]
            self._lever_estimate_valid=True
            self._lever_alignment_pending=False
            self._lever_fit_residual=residual
            self._lever_observation_state=(True,False,received,joint_stamp,None)
            if self._phase=='LATCH_CLEAR_PUSH' and self._raw_contact_panel is not None:
                panel_time,panel_angle=self._raw_contact_panel
                if (abs(panel_time-joint_time)<=.1 and
                        (not self._observed_latch_geometry or
                         panel_time>self._observed_latch_geometry[-1][0])):
                    translation=transform.transform.translation
                    c,s=math.cos(float(yaw)),math.sin(float(yaw))
                    fixed_x=translation.x+c*tool[0]-s*tool[1]
                    fixed_y=translation.y+s*tool[0]+c*tool[1]
                    self._observed_latch_geometry.append(
                        (panel_time,panel_angle,float(fixed_x),float(fixed_y)))
            if joint_time-self._articulation_log_stamp>=.1:
                self._articulation_log_stamp=joint_time
                self._feedback_event('articulation_estimate',measurement_stamp=joint_time,
                    door_angle_rad=self._sim_door_position,lever_angle_rad=angle,
                    residual_rad=residual,source=source,rigid_grasp_assumed=True,
                    encoder_received_wall_sec=received,base_pose_measurement_stamp=pose_time,
                    arm_positions_rad=joints,
                    finger_positions_m=[fingers.get('gripper_left_joint'),fingers.get('gripper_right_joint')])
        except (KeyError,ValueError,TypeError,TransformException) as exc:
            self._lever_estimate_valid=False
            self._lever_alignment_pending=(isinstance(exc,TransformException) or
                str(exc)=='base_pose_not_aligned_with_joint_measurement')
            self._lever_invalid_reason=str(exc)
            self._lever_observation_state=(False,self._lever_alignment_pending,None,None,str(exc))

    def _camera_rotation_fresh(self):
        sample=self._depth_confirmation
        return (sample is not None and -.02<=self._sim_time_sec()-sample[0]<=.25
                and fresh_sample(sample[1],time.monotonic(),3.))

    def _feedback_opening_complete(self,delta,min_delta):
        if not self._lever_estimate_valid:
            return False
        samples=confirmed_plane_opening(list(self._plane_confirmations),self._sim_door_initial_position,
            min_delta,self._sim_time_sec(),time.monotonic())
        if samples is None:
            return False
        self._feedback_event('opening_sensor_reconfirmed',door_delta_rad=delta,
            lower_bound_rad=min(abs(sample['angle']-self._sim_door_initial_position)-sample['uncertainty']
                                for sample in samples),
            sensors=[sample['sensor'] for sample in samples],
            measurement_stamps=[sample['stamp'] for sample in samples])
        return True

    def _feedback_progress_angle(self):
        samples = [sample for sample in self._control_plane_confirmations
                   if fresh_sample(sample['received'], time.monotonic(), 3.)]
        return confirmed_angle_progress(samples, self._sim_door_initial_position,
                                        self._sim_time_sec())

    def _feedback_progress_interval(self):
        now, wall = self._sim_time_sec(), time.monotonic()
        controls = [sample for sample in self._control_plane_confirmations
                    if fresh_sample(sample['stamp'], now, .5)
                    and fresh_sample(sample['received'], wall, 3.)]
        panels = [sample for sample in self._plane_confirmations
                  if fresh_sample(sample['stamp'], now, .5)
                  and fresh_sample(sample['received'], wall, 3.)
                  and math.isfinite(sample['uncertainty'])
                  and 0. <= sample['uncertainty'] <= math.radians(2.)]
        paired = []
        for control in controls[-5:]:
            matches = [panel for panel in panels
                       if abs(panel['stamp']-control['stamp']) < .001]
            if matches:
                paired.append((control, max(p['uncertainty'] for p in matches)))
        if len(paired) < 3 or paired[-1][0]['stamp']-paired[0][0]['stamp'] < .12:
            return None
        angle = confirmed_angle_progress([p[0] for p in paired],
                                         self._sim_door_initial_position, now)
        if angle is None:
            return None
        # Do not shrink independent sensor error by treating nearby frames as
        # independent samples. Reversal must exceed both confidence bounds.
        uncertainty = max(p[1] for p in paired)
        return max(0., angle-uncertainty), angle+uncertainty

    def _feedback_panel_clearance_margin(self, actual):
        samples = [sample for sample in self._plane_confirmations
                   if fresh_sample(sample['stamp'], self._sim_time_sec(), .75)
                   and fresh_sample(sample['received'], time.monotonic(), 3.)]
        if not samples:
            raise ValueError('No fresh panel uncertainty for body clearance')
        fingers=self._sim_arm_joint_positions
        widths=[fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.)]
        return max(grasp_anchored_panel_margin(actual,PIPER_BODY_OUTLINE,
            sample['uncertainty'],widths,self._feedback_config['tool_tolerance_m'])
            for sample in samples)

    def _feedback_panel_barrier_geometry(self, actual, normal):
        now, wall = self._sim_time_sec(), time.monotonic()
        samples = {sample['stamp']: sample for sample in self._plane_confirmations
                   if fresh_sample(sample['stamp'], now, .5)
                   and fresh_sample(sample['received'], wall, 3.)}
        selected=qualified_panel_support(samples.values())
        if (len(selected) < 3 or selected[-1]['stamp']-selected[0]['stamp'] < .12
                or now-selected[-1]['stamp'] > .25):
            # Camera callbacks and the control timer need not arrive in order.
            # A support window still being acquired is not a contradictory
            # plane. Stop motion; the existing bounded freshness guard applies.
            self._publish_sim_cmd_vel(0., 0.)
            raise SceneObservationPending('Independent panel barrier awaits distinct fresh observations')
        _, pose = self._feedback_base_pose_sample()
        c, s = math.cos(pose[2]), math.sin(pose[2])
        inverse = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        origin = np.r_[pose[:2], 0.]
        centers = [inverse@(np.array(sample['center'])-origin) for sample in selected]
        normals = [inverse@np.array(sample['normal']) for sample in selected]
        anchor, plane, margin = observed_panel_barrier_geometry(
            actual, normal, PIPER_BODY_OUTLINE, centers, normals,
            [sample['stamp'] for sample in selected],
            [sample['uncertainty'] for sample in selected],
            [sample['residual'] for sample in selected], now,
            self._feedback_config['tool_tolerance_m'],
            range_uncertainties=[sample.get('range_uncertainty') for sample in selected])
        self._feedback_event('independent_panel_barrier',
            panel_anchor_base_xyz=anchor.tolist(), panel_normal_base_xyz=plane.tolist(),
            hand_to_panel_offset_m=float((anchor-actual)@plane),
            margin_m=margin, measurement_stamps=[sample['stamp'] for sample in selected],
            sources=[sample['sensor'] for sample in selected],
            raw_range_residuals_m=[sample['residual'] for sample in selected],
            plane_offset_uncertainties_m=[sample.get('range_uncertainty') for sample in selected],
            basis='observed_panel_plane_with_range_angle_and_age_reserves')
        return anchor, plane, margin

    def _fingers_released(self):
        fingers=self._sim_arm_joint_positions
        half=self._sim_gripper_release_half_width
        return (fingers.get('gripper_left_joint',0.) >= half
                and -fingers.get('gripper_right_joint',0.) >= half)

    def _feedback_restore_lever_before_release(self):
        self._publish_sim_cmd_vel(0.,0.)
        neutral_stamps=[]
        deadline=SimMotionDeadline(self._sim_time_sec(),time.monotonic(),20.,wall_limit=300.)
        try:
            height=float(self._measured_prepress_height)
            reference,door0,yaw0=self._lever_reference
            if not math.isfinite(height) or self._phase!='RETURN_LEVER':
                raise ValueError('Neutral return requires the measured pre-press baseline')
            while rclpy.ok():
                if not self._feedback_fresh() or not self._lever_estimate_valid:
                    raise ValueError('Lost robot observations during neutral return')
                failure=deadline.failure(self._sim_time_sec(),time.monotonic())
                if failure:
                    raise ValueError(failure)
                if abs(self._sim_door_position-self._sim_door_initial_position) < self._sim_door_contact_min_angle:
                    raise ValueError('Observed door fell below full-open angle during neutral return')
                fingers=self._sim_arm_joint_positions
                if not all(.002<v<=.036 for v in (
                        fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.))):
                    raise ValueError('Opposed grasp lost before neutral lever return')
                actual=self._feedback_tool()
                if actual is None:
                    raise ValueError('Missing measured tool during neutral return')
                lever=self._sim_lever_position-self._sim_lever_initial_position
                lift=height-float(actual[2])
                if abs(lift)>self._feedback_config['press_max_travel_m']:
                    raise ValueError('Neutral return exceeds measured press travel envelope')
                if abs(lever)<=.04 and abs(lift)<=.005:
                    stamp=self._feedback_joint_msg_stamp
                    if stamp is not None and (not neutral_stamps or stamp!=neutral_stamps[-1]):
                        neutral_stamps.append(stamp)
                    if len(neutral_stamps)>=3:
                        first=neutral_stamps[0][0]+neutral_stamps[0][1]*1.e-9
                        last=neutral_stamps[-1][0]+neutral_stamps[-1][1]*1.e-9
                        if last-first>=.16:
                            self._feedback_event('neutral_lever_return_verified',
                                estimated_roll_rad=lever,measured_height_error_m=lift,
                                own_encoder_stamps=neutral_stamps,rigid_grasp_assumed=True)
                            return True
                    time.sleep(.025)
                    continue
                neutral_stamps.clear()
                next_roll=math.copysign(max(0.,abs(lever)-.035),lever)
                self._lever_return_rotation=measured_wrist_rotation(reference,
                    self._sim_door_position-door0,self._feedback_base_yaw()-yaw0,next_roll)
                # No lever pivot/radius is supplied. Follow measured lateral
                # compliance and bound each lift toward the observed baseline.
                goal=actual.copy()
                goal[2]+=float(np.clip(lift,-.003,.003))
                if not self._feedback_move_tool(goal,'held_lever_return'):
                    return False
                self._feedback_event('neutral_lever_return_step',estimated_roll_rad=lever,
                    requested_roll_rad=next_roll,measured_height_error_m=lift,
                    requested_tool_xyz=goal.tolist())
        except (ValueError,AttributeError,TransformException) as exc:
            return self._feedback_stop('Neutral lever return: '+str(exc))
        finally:
            self._lever_return_rotation=None
            self._publish_sim_cmd_vel(0.,0.)
        return self._feedback_stop('Neutral lever return interrupted')

    def _wait_for_sim_gripper_release(self):
        if self._phase != 'RELEASE_HANDLE':
            return super()._wait_for_sim_gripper_release()
        self._publish_sim_cmd_vel(0.,0.)
        deadline=SimMotionDeadline(self._sim_time_sec(),time.monotonic(),12.,wall_limit=90.)
        pending=None
        previous=None
        unchanged_since=self._sim_time_sec()
        try:
            while rclpy.ok():
                if not self._feedback_fresh():
                    return self._feedback_stop('Lost observations during unloaded release')
                if self._fingers_released():
                    self._feedback_event('gripper_release_verified',
                        left_m=self._sim_arm_joint_positions['gripper_left_joint'],
                        right_m=self._sim_arm_joint_positions['gripper_right_joint'])
                    return True
                failure=deadline.failure(self._sim_time_sec(),time.monotonic())
                if failure:
                    return self._feedback_stop('Unloaded release: '+failure)
                fingers=self._sim_arm_joint_positions
                openings=np.array([fingers.get('gripper_left_joint',0.),
                                   -fingers.get('gripper_right_joint',0.)])
                if previous is None or np.max(abs(openings-previous)) > .001:
                    previous=openings.copy()
                    unchanged_since=self._sim_time_sec()
                # One jaw fully open means opposing compression has ended,
                # not that release succeeded. Withdraw only while that remains
                # measured; the unchanged full-aperture test is still mandatory.
                partial=(np.max(openings) >= self._sim_gripper_release_half_width
                         and np.min(openings) > .002)
                if partial and self._sim_time_sec()-unchanged_since >= .35:
                    if pending is None:
                        joints=[fingers[n] for n in PIPER_JOINT_NAMES]
                        span=float(np.sum(openings))
                        cap=released_grip_clearance(self._piper_kinematics.tool_front_extent_m,
                            self._piper_kinematics.tool_contact_offset_m,span,
                            self._feedback_config['tool_tolerance_m'])
                        # A jaw still surrounding the lever is not a released
                        # wrist: withdraw axially before allowing any yaw turn.
                        _,_,path=plan_released_retreat(self._piper_kinematics,joints,cap,
                                                     allow_rotation=False)
                        # Keep the phase's original clocks. Its parent budget
                        # must include the bounded 8 s / 90 s child moves.
                        deadline.sim_limit+=8.*len(path)
                        deadline.wall_limit+=90.*len(path)
                        pending=iter(path)
                        self._feedback_joint_bias=np.zeros(6)
                        self._feedback_event('partial_release_unload_planned',
                            measured_apertures_m=openings.tolist(),maximum_withdrawal_m=cap,
                            base_motion=False,success_not_assumed=True,
                            wrist_rotation_before_full_release=False,
                            bounded_child_steps=len(path),phase_sim_budget_sec=deadline.sim_limit,
                            phase_wall_budget_sec=deadline.wall_limit)
                    step=next(pending,None)
                    if step is None:
                        return self._feedback_stop('Finger release still blocked at geometric withdrawal bound')
                    point,rotation,_=step
                    self._feedback_released_rotation=rotation.copy()
                    if not self._feedback_move_tool(point,'unload_finger_release'):
                        return False
                    self._feedback_event('partial_release_unload_step',requested_xyz=point.tolist(),
                                         measured_apertures_m=openings.tolist())
                time.sleep(.025)
        except (ValueError,TransformException) as exc:
            return self._feedback_stop('Unloaded release: '+str(exc))
        finally:
            self._publish_sim_cmd_vel(0.,0.)
        return self._feedback_stop('Unloaded release interrupted')

    def _feedback_retract_released_handle(self,grip_span):
        self._publish_sim_cmd_vel(0.,0.)
        if not self._fingers_released() or not self._feedback_fresh():
            return self._feedback_stop('Released wrist retreat requires measured open fingers and fresh observations')
        try:
            clearance=released_grip_clearance(self._piper_kinematics.tool_front_extent_m,
                self._piper_kinematics.tool_contact_offset_m,grip_span,
                self._feedback_config['tool_tolerance_m'])
            joints=[self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES]
            origin,normal,path=plan_released_retreat(self._piper_kinematics,joints,clearance)
            self._feedback_joint_bias=np.zeros(6)
            for point,rotation,_ in path:
                if not self._fingers_released() or not self._feedback_fresh():
                    return self._feedback_stop('Release evidence lost before wrist retreat step')
                self._feedback_released_rotation=rotation.copy()
                if not self._feedback_move_tool(point,'released_wrist_retreat'):
                    return False
            final=self._feedback_tool()
            if final is None or (origin-final)@normal < clearance-.002:
                return self._feedback_stop('Measured wrist retreat clearance not reached')
            self._feedback_event('released_handle_clearance_verified',measured_grip_span_m=grip_span,
                requested_clearance_m=clearance,measured_clearance_m=float((origin-final)@normal))
            # Unwinding while the open jaws still surround the lever can hook
            # its return motion. First verify measured axial clearance.
            return self._feedback_relax_released_wrist()
        except (ValueError,TransformException) as exc:
            return self._feedback_stop('Released wrist retreat: '+str(exc))

    def _feedback_relax_released_wrist(self):
        """Undo the measured press posture before axial hand withdrawal."""
        self._publish_sim_cmd_vel(0.,0.)
        try:
            actual=self._feedback_tool()
            if actual is None or not self._fingers_released() or not self._feedback_fresh():
                raise ValueError('Open fingers and fresh encoders are required for wrist unloading')
            measured_height=float(getattr(self,'_measured_prepress_height',float('nan')))
            observed_height=float(getattr(self,'_observed_prepress_height',float('nan')))
            if not all(math.isfinite(value) for value in (measured_height,observed_height)):
                raise ValueError('Missing measured and observed pre-press heights')
            height=max(measured_height,observed_height)
            lift=max(0.,height-float(actual[2]))
            if not math.isfinite(height) or lift > self._feedback_config['press_max_travel_m']:
                raise ValueError('Measured pre-press height exceeds the unloading bound')
            door0,yaw0,reference=self._feedback_contact_orientation_reference
            door_delta=self._sim_door_position-door0
            base_delta=self._feedback_base_yaw()-yaw0
            _,rotation=self._piper_kinematics.forward([
                self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
            roll,_=lever_from_wrist(reference,rotation,door_delta,base_delta)
            count=max(1,int(math.ceil(max(lift/.006,abs(roll)/.04))))
            self._feedback_joint_bias=np.zeros(6)
            for index in range(1,count+1):
                if not self._fingers_released() or not self._feedback_fresh():
                    raise ValueError('Observation or finger opening lost during wrist unloading')
                fraction=index/count
                goal=actual.copy(); goal[2]+=fraction*lift
                self._feedback_released_rotation=measured_wrist_rotation(
                    reference,self._sim_door_position-door0,self._feedback_base_yaw()-yaw0,
                    roll*(1.-fraction))
                if not self._feedback_move_tool(goal,'released_wrist_unload'):
                    return False
            measured=self._feedback_tool()
            _,rotation=self._piper_kinematics.forward([
                self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
            error=np.linalg.norm(self._piper_kinematics._rotation_error(
                rotation,self._feedback_released_rotation))
            if measured is None or measured[2] < height-.002 or error > .06:
                raise ValueError('Measured wrist has not unloaded to the pre-press posture')
            self._feedback_event('released_wrist_unloaded',restored_reference_height_m=height,
                observed_handle_height_m=observed_height,measured_prepress_tool_height_m=measured_height,
                measured_lift_m=float(measured[2]-actual[2]),measured_orientation_error_rad=float(error))
            return True
        except (ValueError,KeyError,TransformException) as exc:
            return self._feedback_stop('Released wrist unloading: '+str(exc))
        finally:
            self._publish_sim_cmd_vel(0.,0.)

    def _feedback_press_increment(self, goal, actual, step):
        cfg = self._feedback_config
        return observed_press_increment(goal, actual, step,
            cfg['tracking_limit_m'], cfg['tool_tolerance_m'])

    def _feedback_press(self,handle_pos):
        try:
            tool,rotation=self._piper_kinematics.forward([
                self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
            yaw=self._feedback_base_yaw()
            if not self._feedback_fresh():
                return self._feedback_stop('Cannot initialize observed contact baseline')
            height=float(handle_pos.point.z)
            if (not math.isfinite(height)
                    or abs(height-float(tool[2])) > .005):
                return self._feedback_stop('Measured grasp height does not match the observed handle')
            self._lever_reference=(rotation.copy(),self._sim_door_position,yaw)
            self._measured_prepress_height=float(tool[2])
            self._observed_prepress_height=height
        except (ValueError,KeyError,TransformException) as exc:
            return self._feedback_stop('Contact baseline: '+str(exc))
        # Zero is a relative encoder baseline at grasp, not an environmental reading.
        self._sim_lever_position=0.
        self._sim_lever_initial_position=0.
        self._feedback_lever_samples=[]
        self._feedback_lever_stamp=self._feedback_arm_stamp
        self._lever_tracking=True
        self._lever_estimate_valid=True
        self._lever_observation_state=(True,False,self._feedback_arm_stamp,
                                       self._feedback_joint_msg_stamp,None)
        self._feedback_event('encoder_lever_baseline',
            assumption='rigid grasp; latch release still requires observed door motion',
            observed_handle_height_m=height,measured_tool_height_m=float(tool[2]),
            reference_rotation=rotation.tolist())
        return super()._feedback_press(handle_pos)

    def _feedback_base_yaw(self):
        # TF and /clock arrive independently. Request a common past/current
        # timestamp instead of treating a just-arrived newer TF as stale.
        requested=Time(seconds=self._sim_time_sec())
        began=time.monotonic(); waiting=False; reason='base TF unavailable'
        while rclpy.ok():
            try:
                transform=self._tf_buffer.lookup_transform(self._contact_reference_frame,self._manipulation_frame,Time())
                latest=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
                if latest>requested.nanoseconds*1.e-9:
                    transform=self._tf_buffer.lookup_transform(self._contact_reference_frame,self._manipulation_frame,requested)
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

    def _feedback_follow_contact(self):
        # A shifted visual point is never a new grip command. During proven
        # self-occlusion, use the existing encoder mode with all its stop checks.
        if self._feedback_follow_mode == 'vision' and self._feedback_encoder_contact_follow:
            sample = self._feedback_live_contact_observation()
            reference = self._feedback_follow_reference
            actual = self._feedback_tool()
            if (sample is None and reference is not None and actual is not None
                    and self._feedback_press_verified and self._lever_estimate_valid):
                fingers=self._sim_arm_joint_positions
                opposed=all(.002<v<=.036 for v in (
                    fingers.get('gripper_left_joint',0.),-fingers.get('gripper_right_joint',0.)))
                try:
                    _,rotation=self._piper_kinematics.forward([fingers[n] for n in PIPER_JOINT_NAMES])
                    camera=self._tf_buffer.lookup_transform(self._manipulation_frame,
                        'camera_color_optical_frame',Time()).transform.translation
                    retained=PointStamped()
                    retained.header.frame_id=self._contact_reference_frame
                    retained.point.x,retained.point.y,retained.point.z=map(float,reference[0])
                    target=self._transform_handle_to_manipulation_frame(retained).point
                    occluded=gripper_occludes_ray([camera.x,camera.y,camera.z],
                        [target.x,target.y,target.z],actual,rotation,
                        self._piper_kinematics.tool_contact_offset_m,
                        self._piper_kinematics.tool_front_extent_m,
                        self._feedback_config['tool_tolerance_m'])
                    if opposed and occluded and self._feedback_fresh():
                        self._feedback_event('missing_visual_self_occlusion_confirmed',
                            ray_occluded=True,opposed_jaws=True,
                            basis='own_encoder_gripper_geometry_and_fresh_panel_observation')
                        return self._feedback_begin_encoder_follow()
                except (TransformException,AttributeError,ValueError):
                    pass
            if sample is not None and reference is not None and actual is not None:
                p = sample[1].point
                displacement = np.linalg.norm(np.array([p.x, p.y])-reference[0][:2])
                if displacement > self._feedback_config['follow_max_shift_m']:
                    try:
                        local = self._transform_handle_to_manipulation_frame(sample[1]).point
                        _, rotation = self._piper_kinematics.forward([
                            self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
                        camera = self._tf_buffer.lookup_transform(
                            self._manipulation_frame, 'camera_color_optical_frame', Time()).transform.translation
                        retained = PointStamped()
                        retained.header.frame_id = self._contact_reference_frame
                        retained.point.x, retained.point.y, retained.point.z = map(float, reference[0])
                        target = self._transform_handle_to_manipulation_frame(retained).point
                        geometry = (self._piper_kinematics.tool_contact_offset_m,
                                    self._piper_kinematics.tool_front_extent_m,
                                    self._feedback_config['tool_tolerance_m'])
                        self_hit = inside_gripper_envelope([local.x, local.y, local.z], actual, rotation, *geometry)
                        occluded = gripper_occludes_ray([camera.x, camera.y, camera.z],
                            [target.x, target.y, target.z], actual, rotation, *geometry)
                        if (self_hit or occluded) and self._lever_estimate_valid:
                            self._feedback_event('visual_self_occlusion_rejected', shift_m=float(displacement),
                                                 self_hit=bool(self_hit), ray_occluded=bool(occluded))
                            return self._feedback_begin_encoder_follow()
                    except (TransformException, AttributeError, ValueError):
                        pass
        return super()._feedback_follow_contact()

    def _feedback_neutral_return_error(self,actual,goal):
        sample=self._observed_panel_segment
        if (sample is None or not fresh_sample(sample[0],self._sim_time_sec(),.25)
                or not fresh_sample(sample[1],time.monotonic(),3.)
                or sample[2].get('reference_frame')!=self._contact_reference_frame
                or 'angle_rad' not in sample[2]
                or sample[2].get('angle_measurement_valid') is False):
            raise NeutralPanelObservationPending('Neutral return requires a fresh observed panel normal')
        normal=np.asarray(sample[2]['normal'],float)[:2]
        yaw=self._feedback_base_yaw()
        c,s=math.cos(yaw),math.sin(yaw)
        normal=np.array([[c,s],[-s,c]])@normal
        normal/=np.linalg.norm(normal)
        return neutral_return_task_error(actual,goal,normal,self._feedback_config['follow_step_m'])

    def _feedback_move_tool(self,goal,stage):
        # Loaded contact uses tangential relaxation, not free-space bias buildup.
        if stage=='encoder_contact_follow' and getattr(self,'_feedback_full_handle_follow',False):
            return ManipulationNode._feedback_move_tool(self,goal,stage)
        # Original bounded servo, with camera-based pre-contact disturbance guard.
        # The preserved version assumes an environmental lever sensor even here.
        if not self._feedback_command_point(goal,stage): return False
        deadline=SimMotionDeadline(self._sim_time_sec(),time.monotonic(),8.,wall_limit=90.)
        stable_since=None; next_sample=self._sim_time_sec(); previous_sample=None
        neutral_wait=None
        while rclpy.ok():
            failure=deadline.failure(self._sim_time_sec(),time.monotonic())
            if failure: return self._feedback_stop('Approach step: '+failure)
            actual=self._feedback_tool()
            if actual is None: return self._feedback_stop('Lost feedback while approaching handle')
            cartesian_error=float(np.linalg.norm(actual-goal))
            error=cartesian_error
            tangent_shift=None
            if stage=='held_lever_return':
                try:
                    error,tangent_shift=self._feedback_neutral_return_error(actual,goal)
                except NeutralPanelObservationPending as exc:
                    self._publish_sim_cmd_vel(0.,0.)
                    if neutral_wait is None:
                        neutral_wait=(self._sim_time_sec(),time.monotonic())
                        targets={n:self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES}
                        self._command_sim_arm_targets('neutral_panel_wait_hold',targets)
                        self._feedback_event('neutral_panel_reobservation',reason=str(exc),
                                             arm_motion_held=True,original_deadline_preserved=True)
                    if (self._sim_time_sec()-neutral_wait[0]>2. or
                            time.monotonic()-neutral_wait[1]>30.):
                        return self._feedback_stop('Neutral panel unavailable after stopped reobservation')
                    stable_since=None
                    time.sleep(.025)
                    continue
                except (ValueError,TransformException) as exc:
                    return self._feedback_stop('Neutral return task constraint: '+str(exc))
                if neutral_wait is not None:
                    self._feedback_event('neutral_panel_recovered',
                        waited_sim_sec=self._sim_time_sec()-neutral_wait[0])
                    if not self._feedback_command_point(goal,stage):
                        return False
                    neutral_wait=None
            tolerance=self._feedback_config['tool_tolerance_m']
            if stage in ('observed_pre_grasp','feedback_approach'):
                tolerance=min(tolerance,.005)
            if stage in ('unload_finger_release','released_wrist_unload','released_wrist_retreat',
                         'held_lever_return'):
                tolerance=min(tolerance,.002)
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
            orientation_tolerance=.03 if stage=='held_lever_return' else .10
            if self._sim_time_sec()>=next_sample:
                self._feedback_event('approach_sample',stage=stage,actual_tool_xyz=actual.tolist(),
                    requested_xyz=np.asarray(goal).tolist(),error_m=error,
                    full_cartesian_error_m=cartesian_error,
                    bounded_panel_tangent_shift_m=tangent_shift,
                    orientation_error_rad=orientation_error,joint_tracking_error_rad=float(joint_error))
                if (previous_sample is not None and np.linalg.norm(actual-previous_sample)<.002
                        and (error>tolerance or orientation_error>orientation_tolerance)):
                    measured=np.array([self._sim_arm_joint_positions[n] for n in PIPER_JOINT_NAMES])
                    gain=1. if stage=='held_lever_return' else .5
                    candidate=self._feedback_joint_bias+gain*(self._feedback_ideal_arm_target-measured)
                    if np.max(np.abs(candidate))>self._feedback_config['joint_bias_limit_rad']:
                        return self._feedback_stop('Joint bias correction limit exceeded')
                    self._feedback_joint_bias=candidate
                    if not self._feedback_command_point(goal,stage): return False
                    self._feedback_event('servo_correction',joint_bias_rad=candidate.tolist(),
                        measured_error_m=error,orientation_error_rad=orientation_error)
                previous_sample=actual.copy()
                next_sample=self._sim_time_sec()+(.25 if stage=='held_lever_return' else .5)
            if stage=='observed_pre_grasp' and not self._pregrasp_observation_consistent():
                return self._feedback_stop('Observed door or handle moved during pre-grasp')
            if error<=tolerance and orientation_error<=orientation_tolerance:
                stable_since=stable_since or self._sim_time_sec()
                if self._sim_time_sec()-stable_since>=.35:
                    self._feedback_event('approach_reached',stage=stage,actual_tool_xyz=actual.tolist(),
                        requested_xyz=np.asarray(goal).tolist(),error_m=error)
                    return True
            else: stable_since=None
            time.sleep(.025)
        return self._feedback_stop('Approach failed to converge; no blind advance')

    def _move_to_home(self):
        if not (self._feedback_contact and self._hold_handle_during_base_open
                and self._phase == 'RETURN_HOME'):
            return super()._move_to_home()
        # The held-open sequence has already released, withdrawn and folded
        # the arm. Verify that measured pose, without requiring a visible door
        # or issuing another trajectory after the camera faces away from it.
        self._publish_sim_cmd_vel(0., 0.)
        began = time.monotonic()
        began_sim = self._sim_time_sec()
        while rclpy.ok():
            joints = dict(self._sim_arm_joint_positions)
            stamp = getattr(self, '_feedback_joint_msg_stamp', None)
            received = self._feedback_arm_stamp
            now, sim_now = time.monotonic(), self._sim_time_sec()
            measured = None if stamp is None else stamp[0] + stamp[1] * 1.e-9
            fresh = (fresh_sample(received, now, self._feedback_config['max_age_sec'])
                     and fresh_sample(measured, sim_now, .15))
            if fresh:
                if not self._fingers_released():
                    return self._feedback_stop('Open fingers not confirmed at final stow')
                values = np.asarray([joints.get(n, float('nan')) for n in PIPER_JOINT_NAMES])
                if not np.all(np.isfinite(values)) or np.max(abs(values - PIPER_STOW)) > .070:
                    return self._feedback_stop('Measured final arm stow was not confirmed')
                self._feedback_event('home_verified',
                    verification='fresh robot encoders after released recovery',
                    max_joint_error_rad=float(np.max(abs(values - PIPER_STOW))),
                    joint_sample_age_sec=sim_now-measured,
                    door_angle_required=False,additional_arm_motion=False)
                return True
            if now-began >= 2. or sim_now-began_sim >= 2.:
                return self._feedback_stop('Fresh robot encoders unavailable at final stow')
            time.sleep(.01)
        return self._feedback_stop('Final arm stow verification interrupted')

    def _feedback_fresh(self):
        began=time.monotonic(); began_sim=self._sim_time_sec(); paused=False
        frozen_stream_wait=False
        # Before any contact motion, collect the stationary observed reference
        # and closed plane. This is acquisition time, never a fresher old sample.
        initialization=(self._phase=='LOCALIZE_HANDLE' and not self._lever_tracking)
        acquisition_limit=4. if initialization else 2.
        while True:
            maximum=self._feedback_config['max_age_sec']
            arm_received=self._feedback_arm_stamp
            stamp=self._feedback_door_msg_stamp
            door_received=self._feedback_door_stamp
            agreement=(not self._lever_tracking or self._encoder_panel_agreement)
            lever_state=getattr(self,'_lever_observation_state',None)
            if lever_state is None:
                lever_state=(getattr(self,'_lever_estimate_valid',False),
                    getattr(self,'_lever_alignment_pending',False),
                    self._feedback_lever_stamp,None,getattr(self,'_lever_invalid_reason',None))
            lever_valid,lever_pending,lever_received,_,lever_reason=lever_state
            # Read original receipts before reading time. A concurrent callback
            # must not appear "from the future" merely because it ran mid-check.
            sim_now=self._sim_time_sec(); now=time.monotonic()
            measured=None if stamp is None else stamp[0]+stamp[1]*1.e-9
            arm=fresh_sample(arm_received,now,maximum)
            door=(self._angle_freshness.valid(measured,door_received,sim_now,now) and agreement)
            lever=(not self._lever_tracking or (lever_valid and fresh_sample(lever_received,now,maximum)))
            if arm and door and lever:
                if paused: self._feedback_event('sensor_stream_recovered',waited_real_sec=now-began)
                return True
            self._publish_sim_cmd_vel(0.,0.)
            if not paused:
                self._feedback_measurement_pause_generation+=1
                self._feedback_event('sensor_stream_wait',arm_fresh=arm,door_fresh=door,
                                     lever_estimate_fresh=lever,phase=self._phase,
                                     angle_age_sec=None if measured is None else self._sim_time_sec()-measured,
                                     lever_invalid_reason=lever_reason)
                paused=True
            # A stopped panel reacquisition may take several real seconds on
            # a slow simulator. Never increase measurement freshness or move
            # while waiting; missing robot encoders still fails promptly.
            joint_stamp=getattr(self,'_feedback_joint_msg_stamp',None)
            joint_time=None if joint_stamp is None else joint_stamp[0]+joint_stamp[1]*1.e-9
            eligible_encoder=fresh_sample(joint_time,self._sim_time_sec(),.15)
            # A just-arrived encoder packet can be ahead of /clock. A fresh
            # original earlier packet may still qualify for stopped TF recovery.
            if not eligible_encoder:
                eligible_encoder=any(
                    fresh_sample(sample[0][0]+sample[0][1]*1.e-9,self._sim_time_sec(),.15)
                    and fresh_sample(sample[1],now,maximum)
                    for sample in list(getattr(self,'_lever_encoder_history',())))
            alignment_pending=((lever_pending or lever_valid and not lever) and arm
                and eligible_encoder)
            sim_elapsed=sim_now-began_sim
            # A simulator-wide pause is not a new sensor observation. Keep the
            # chassis stopped and allow bounded delivery after /clock resumes.
            if (not frozen_stream_wait and now-began>=2. and abs(sim_elapsed)<=1.e-9
                    and not arm and not door and (not self._lever_tracking or lever_valid)):
                frozen_stream_wait=True
                self._feedback_event('frozen_clock_stream_wait',phase=self._phase,
                    waited_real_sec=now-began,maximum_real_sec=5.,
                    maximum_resume_sim_sec=.15)
            frozen_grace=(frozen_stream_wait and 0.<=sim_elapsed<=.15
                and (not self._lever_tracking or lever_valid))
            missing_stream_limit=5. if frozen_grace else 2.
            if (not rclpy.ok() or (not arm or not lever and not alignment_pending) and now-began>=missing_stream_limit
                    or now-began>=30. or self._sim_time_sec()-began_sim>=acquisition_limit):
                self._feedback_event('sensor_stream_recovery_failed',phase=self._phase,
                    arm_fresh=arm,door_fresh=door,lever_estimate_fresh=lever,
                    alignment_pending=alignment_pending,eligible_own_encoder=eligible_encoder,
                    waited_real_sec=now-began,waited_sim_sec=self._sim_time_sec()-began_sim,
                    stopped_acquisition_limit_sim_sec=acquisition_limit,
                    frozen_clock_grace=frozen_grace,missing_stream_limit_real_sec=missing_stream_limit,
                    lever_estimate_valid=lever_valid,lever_receipt_age_sec=None if lever_received is None else now-lever_received,
                    lever_invalid_reason=lever_reason)
                return False
            time.sleep(.01)

    def _feedback_event(self,event,**values):
        if getattr(self,'_r34_reference_active',False) and event=='handle_held_open_verified':
            event='r34_reference_threshold_reached_unconfirmed'
        payload=dict(event=event,sim_time_sec=self._sim_time_sec(),
            feedback_source='door:RGB-D plus observed LiDAR panel; lever:measured wrist FK under grasp assumption; arm:robot encoders',
            environmental_joint_feedback_used=False,
            contact_reference_frame=getattr(self,'_contact_reference_frame','odom'),
            motion_policy=getattr(self,'_contact_motion_policy','measured'),**values)
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
