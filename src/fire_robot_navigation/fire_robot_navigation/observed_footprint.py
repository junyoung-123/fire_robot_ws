"""Short chassis sweeps supported by live scans and an observed costmap."""
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def historical_ray_support(points, current_odom, current_stamp, history, margin=.08,
                           max_age_sec=2.):
    """Two recent, independent measured views may cover a current blind ray.

    Poses are timestamped wheel odometry, not SLAM map corrections. Unknown or
    obstacle-shadowed rays never certify free space; no recursive map filling.
    """
    points=np.asarray(points,float)
    current_odom=np.asarray(current_odom,float)
    if (current_odom.shape!=(3,) or not np.isfinite(current_odom).all()
            or not math.isfinite(current_stamp)
            or not math.isfinite(max_age_sec) or not 0.<max_age_sec<=30.):
        return np.zeros(len(points),bool)
    c,s=math.cos(current_odom[2]),math.sin(current_odom[2])
    world=points@np.array([[c,s],[-s,c]])+current_odom[:2]
    support=np.zeros(len(points),int)
    used=[]
    for stamp,pose,sensor,scan in reversed(history):
        # Separation prevents repeated callbacks from becoming extra evidence.
        age=current_stamp-stamp
        separation = .12 if age<=2. else 1.5
        if not .08<=age<=max_age_sec or any(abs(stamp-t)<separation for t in used):
            continue
        pose,sensor=np.asarray(pose,float),np.asarray(sensor,float)
        if pose.shape!=(3,) or sensor.shape!=(3,) or not np.isfinite(np.r_[pose,sensor]).all():
            continue
        extra_margin=0.
        if age>2.:
            # Explicit static-scene mode only. Bound local odometry separation
            # and reserve additional translation/yaw uncertainty for old rays.
            displacement=float(np.linalg.norm(current_odom[:2]-pose[:2]))
            turn=abs(math.atan2(math.sin(current_odom[2]-pose[2]),
                                math.cos(current_odom[2]-pose[2])))
            if displacement>.5 or turn>math.pi/2:
                continue
            extra_margin=.02*displacement+.01*turn
        ranges=np.asarray(scan.ranges,float)
        if len(ranges)<3 or not math.isfinite(scan.angle_increment) or scan.angle_increment<=0.:
            continue
        c,s=math.cos(pose[2]),math.sin(pose[2])
        local=(world-pose[:2])@np.array([[c,-s],[s,c]])-sensor[:2]
        angle=np.arctan2(local[:,1],local[:,0])-sensor[2]
        angle=np.arctan2(np.sin(angle),np.cos(angle))
        index=np.rint((angle-scan.angle_min)/scan.angle_increment).astype(int)
        valid=np.isfinite(ranges)&(ranges>=scan.range_min)&(ranges<=scan.range_max)
        limits=np.where(valid,ranges,np.where(np.isposinf(ranges),scan.range_max,np.nan))
        neighbors=np.clip(index[:,None]+np.array([-1,0,1]),0,len(limits)-1)
        observed=np.min(limits[neighbors],axis=1)
        support+=((index>=0)&(index<len(limits))&np.isfinite(observed)
                  &(np.linalg.norm(local,axis=1)+margin+extra_margin<observed))
        used.append(stamp)
        if np.all(support>=2):
            break
    return support>=2


def arc_pose(speed, angular, elapsed):
    angle = angular * elapsed
    if abs(angular) < 1.e-8:
        return np.array([speed * elapsed, 0., angle])
    return np.array([speed / angular * math.sin(angle),
                     speed / angular * (1. - math.cos(angle)), angle])


class ObservedSweep:
    def __init__(self, scan, sensor_pose, body_bounds, map_pose, grid, margin=.05,
                 free_history=(), odom_pose=None, measurement_stamp=None,
                 free_history_age_sec=2., local_grid=None):
        self.scan, self.sensor = scan, np.asarray(sensor_pose, float)
        self.bounds, self.pose = np.asarray(body_bounds, float), np.asarray(map_pose, float)
        self.grid, self.margin = grid, margin
        self.free_history,self.odom_pose,self.measurement_stamp=free_history,odom_pose,measurement_stamp
        self.free_history_age_sec=free_history_age_sec
        self.local_grid=local_grid
        if (self.bounds.shape != (4,) or not np.isfinite(self.bounds).all()
                or self.bounds[0] >= self.bounds[1] or self.bounds[2] >= self.bounds[3]
                or self.sensor.shape != (3,) or self.pose.shape != (3,)
                or not np.isfinite(self.sensor).all() or not np.isfinite(self.pose).all()
                or not math.isfinite(scan.angle_increment) or scan.angle_increment <= 0
                or len(scan.ranges) < 3
                or scan.angle_increment*(len(scan.ranges)-1) < math.radians(350)):
            raise ValueError('Invalid observed sweep geometry')
        ranges = np.asarray(scan.ranges, float)
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        if valid.sum() < 3:
            raise ValueError('Insufficient measured obstacle support')
        angles = scan.angle_min + np.arange(len(ranges))*scan.angle_increment + self.sensor[2]
        self.obstacles = self.sensor[:2] + np.column_stack((
            ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid])))
        self.ray_limits = np.where(valid, ranges, np.where(np.isposinf(ranges), scan.range_max, np.nan))
        # Sample the expanded body, not just its centre or a radial sector.
        lo = self.bounds[[0, 2]] - margin
        hi = self.bounds[[1, 3]] + margin
        xs, ys = (np.linspace(lo[i], hi[i], math.ceil((hi[i]-lo[i])/.025)+1) for i in range(2))
        xx, yy = np.meshgrid(xs, ys)
        self.samples = np.column_stack((xx.ravel(), yy.ravel()))
        self.start_lo, self.start_hi = lo, hi

    def clearance(self, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        points = (self.obstacles-pose[:2]) @ np.array([[c, -s], [s, c]])
        delta = points - np.clip(points, self.bounds[[0, 2]], self.bounds[[1, 3]])
        return float(np.min(np.linalg.norm(delta, axis=1)))

    def observed_free(self, points, *, padded=False):
        # Swept samples already include the full chassis margin. Applying the
        # same radial margin again creates a second, inconsistent inflation.
        ray_margin = 0. if padded else self.margin
        delta = points-self.sensor[:2]
        distance = np.linalg.norm(delta, axis=1)
        angle = np.arctan2(delta[:, 1], delta[:, 0])-self.sensor[2]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        indexes = np.rint((angle-self.scan.angle_min)/self.scan.angle_increment).astype(int)
        # Three adjacent valid rays are required; a NaN never clears space.
        neighbors = np.clip(indexes[:, None]+np.array([-1, 0, 1]), 0, len(self.ray_limits)-1)
        neighbor_limits = self.ray_limits[neighbors]
        limits = np.min(neighbor_limits, axis=1)
        in_scan = (indexes >= 0) & (indexes < len(self.ray_limits))
        free = (in_scan
                & np.isfinite(limits) & (distance+ray_margin < limits))
        if free.all():
            return free
        known_limit = np.min(np.where(np.isnan(neighbor_limits), np.inf, neighbor_limits), axis=1)
        known_rays_clear = distance+ray_margin < known_limit
        if self.free_history and self.odom_pose is not None and self.measurement_stamp is not None:
            blind=in_scan & np.isnan(limits) & known_rays_clear
            if blind.any():
                free[blind] |= historical_ray_support(points[blind],self.odom_pose,
                    self.measurement_stamp,self.free_history,margin=ray_margin+.03,
                    max_age_sec=self.free_history_age_sec)
        blind = in_scan & np.isnan(limits) & known_rays_clear
        remembered = self._grid_costs(points, self.pose, self.grid)
        support = (remembered >= 0) & (remembered < 99)
        if self.local_grid is not None and self.odom_pose is not None:
            local = self._grid_costs(points, self.odom_pose, self.local_grid)
            # This opt-in grid starts unknown and is cleared only by scans,
            # never by a saved map or automatic footprint clearing. Odometry
            # keeps its local memory independent of SLAM map corrections.
            support |= local == 0
            free[blind & (local >= 99)] = False
            support &= local < 99
        # 99 is Nav2's inscribed cost: inflation may put it into an UNKNOWN
        # cell without a clearing ray. It cannot prove observed free space.
        return free | (blind & support)

    @staticmethod
    def _grid_costs(points, pose, grid):
        costs = np.full(len(points), -1, dtype=int)
        if grid is None or np.shape(pose)!=(3,) or not np.isfinite(pose).all():
            return costs
        info = grid.info
        q = info.origin.orientation
        values = (info.resolution, q.x, q.y, q.z, q.w,
                  info.origin.position.x, info.origin.position.y)
        if (not all(math.isfinite(v) for v in values) or info.resolution <= 0
                or abs(q.x)+abs(q.y) > 1.e-5 or abs(q.w*q.w+q.z*q.z-1.) > .02
                or len(grid.data) != info.width*info.height):
            return costs
        c, s = math.cos(pose[2]), math.sin(pose[2])
        world = points @ np.array([[c, s], [-s, c]]) + np.asarray(pose)[:2]
        origin = np.array([info.origin.position.x, info.origin.position.y])
        yaw = math.atan2(2*q.w*q.z, 1-2*q.z*q.z)
        c, s = math.cos(yaw), math.sin(yaw)
        ij = np.floor(((world-origin) @ np.array([[c, -s], [s, c]]))/info.resolution).astype(int)
        inside = ((ij[:, 0] >= 0) & (ij[:, 0] < info.width)
                  & (ij[:, 1] >= 0) & (ij[:, 1] < info.height))
        data = np.asarray(grid.data)
        costs[inside] = data[ij[inside, 1]*info.width+ij[inside, 0]]
        return costs

    def safe(self, speed, angular, horizon=1.5, start=None):
        self.last_failure = None
        if not all(math.isfinite(v) for v in (speed, angular, horizon)) or horizon <= 0:
            return False
        if abs(speed) > .12 or abs(angular) > .5:
            return False
        start = np.zeros(3) if start is None else np.asarray(start, float)
        if start.shape != (3,) or not np.isfinite(start).all():
            return False
        c0, s0 = math.cos(start[2]), math.sin(start[2])
        initial_gap = self.clearance(np.zeros(3))
        withdrawing = False
        if initial_gap < .03:
            # A close contact stance may withdraw, but never turn or move
            # further into the obstacle. Actual body intersections still stop.
            withdrawing = (initial_gap > 1.e-6 and -.03 <= speed < 0.
                           and abs(angular) < 1.e-8 and horizon <= 1.5
                           and np.linalg.norm(start) < 1.e-8)
            if not withdrawing:
                self.last_failure = 'HARD_BODY_CLEARANCE'
                return False
        # The contact stage may stop inside the preferred navigation margin.
        # Permit only non-decreasing clearance until that margin is recovered.
        required_gap = min(self.margin, initial_gap)
        for t in np.linspace(0., horizon, max(2, math.ceil(horizon/.1)+1)):
            pose = arc_pose(speed, angular, t)
            pose[:2] = pose[:2] @ np.array([[c0, s0], [-s0, c0]]) + start[:2]
            pose[2] += start[2]
            if self.clearance(pose) < required_gap-1.e-8:
                self.last_failure = 'SWEEP_APPROACHES_OBSTACLE'
                return False
            c, s = math.cos(pose[2]), math.sin(pose[2])
            points = self.samples @ np.array([[c, s], [-s, c]]) + pose[:2]
            newly_entered = np.any((points < self.start_lo-1.e-6) | (points > self.start_hi+1.e-6), axis=1)
            if newly_entered.any() and not self.observed_free(points[newly_entered], padded=True).all():
                self.last_failure = 'UNKNOWN_OR_OCCUPIED_SWEEP'
                return False
        if withdrawing and self.clearance(pose) < initial_gap+.005:
            self.last_failure = 'WITHDRAWAL_DOES_NOT_GAIN_CLEARANCE'
            return False
        return True

    def choose(self, yaw_error, speed=.06):
        self.recovery_path = None
        speed = min(abs(speed), .06)
        start_gap = self.clearance(np.zeros(3))
        if start_gap < .03:
            # A full retreat may reach an unobserved rear strip while a shorter
            # sweep is measured clear. Keep the same clearance-gain predicate.
            for reverse in (-.03,-.015,-.0075):
                if self.safe(reverse, 0.):
                    return reverse, 0., 'OBSERVED_CLEARANCE_INCREASING_WITHDRAWAL'
            return 0., 0., 'NO_OBSERVED_SWEEP'
        # Near a doorway, forward progress must not repeatedly consume the
        # space needed to turn toward the measured return target.
        if abs(yaw_error) > .20:
            for angular in (.24, .12, .08, .04):
                turn = math.copysign(min(angular, abs(yaw_error)/1.5), yaw_error)
                if self.safe(0., turn):
                    return 0., turn, 'OBSERVED_RETURN_HEADING'
        choices = []
        commands=[(v,w) for v in (speed,0.,-speed) for w in (0.,-.12,.12,-.24,.24)]
        for v,w in commands:
            if abs(v)+abs(w) < 1.e-8 or not self.safe(v, w):
                continue
            pose = arc_pose(v, w, 1.5)
            progress = pose[0]*math.cos(yaw_error)+pose[1]*math.sin(yaw_error)
            error = math.atan2(math.sin(yaw_error-pose[2]), math.cos(yaw_error-pose[2]))
            gain = min(.4, self.clearance(pose))-min(.4, start_gap)
            score = 4*progress + 1.2*(abs(yaw_error)-abs(error)) + .8*gain
            score -= .005 if v < 0 else 0.
            choices.append((score, v, w))
        if not any(candidate[1]>0. for candidate in choices):
            # A 9cm look-ahead can be blocked while a smaller observed step
            # exposes new rays. Keep the same footprint, margin and horizon.
            for v in (min(speed,.03),min(speed,.015),0.,-min(speed,.03)):
                for w in (0.,-.04,.04,-.08,.08):
                    if abs(v)+abs(w)<1.e-8 or not self.safe(v,w):
                        continue
                    pose=arc_pose(v,w,1.5)
                    progress=pose[0]*math.cos(yaw_error)+pose[1]*math.sin(yaw_error)
                    error=math.atan2(math.sin(yaw_error-pose[2]),math.cos(yaw_error-pose[2]))
                    gain=min(.4,self.clearance(pose))-min(.4,start_gap)
                    score=4*progress+1.2*(abs(yaw_error)-abs(error))+.8*gain-(.005 if v<0 else 0.)
                    choices.append((score,v,w))
        if not choices:
            return 0., 0., 'NO_OBSERVED_SWEEP'
        _, v, w = max(choices)
        if max(choices)[0] <= 1.e-4:
            recovery = self._lookahead_recovery(yaw_error, speed, start_gap)
            if recovery is not None:
                return *recovery, 'OBSERVED_MULTI_STEP_SWEEP'
            # Do not undo a measured retreat just because moving away from
            # the return target is the only currently clear one-step command.
            return 0., 0., 'NO_OBSERVED_PROGRESS'
        if not any(candidate[1] > 0. for candidate in choices):
            recovery = self._lookahead_recovery(yaw_error, speed, start_gap)
            if recovery is not None:
                return *recovery, 'OBSERVED_MULTI_STEP_SWEEP'
        return v, w, 'OBSERVED_FOOTPRINT_SWEEP'

    def _lookahead_recovery(self, yaw_error, speed, start_gap, *, depth_limit=4,
                            beam_width=10, minimum_progress=.06):
        # A jamb may require a short retreat before a forward arc is possible.
        # Every segment uses the same observed-space and whole-body checks.
        commands = [(v, w) for v in (speed, 0., -speed)
                    for w in (0., -.24, .24) if abs(v)+abs(w) > 1.e-8]
        # Doorway clearances can admit a short arc but reject every 9cm/20deg
        # primitive. Keep those smaller, fully checked steps in the route too,
        # rather than allowing them only in the one-step fallback above.
        fine_speed = min(speed, .03)
        commands += [(v, w) for v in (fine_speed, -fine_speed)
                     for w in (0., -.08, .08)]
        commands += [(0., -.08), (0., .08)]
        commands = list(dict.fromkeys(commands))
        beam = [(0., np.zeros(3), [], 0.)]
        solutions = []
        for depth in range(depth_limit):
            candidates = {}
            for _, start, path, traveled in beam:
                for v, w in commands:
                    if not self.safe(v, w, start=start):
                        continue
                    local = arc_pose(v, w, 1.5)
                    c, s = math.cos(start[2]), math.sin(start[2])
                    end = local.copy()
                    end[:2] = local[:2] @ np.array([[c, s], [-s, c]]) + start[:2]
                    end[2] += start[2]
                    distance = traveled + abs(v)*1.5
                    progress = end[0]*math.cos(yaw_error)+end[1]*math.sin(yaw_error)
                    error = math.atan2(math.sin(yaw_error-end[2]), math.cos(yaw_error-end[2]))
                    gain = min(.4, self.clearance(end))-min(.4, start_gap)
                    score = 4*progress + 1.2*(abs(yaw_error)-abs(error)) + .8*gain - .10*distance
                    item = (score, end, path+[(v, w, end.copy())], distance)
                    key = (round(end[0]/.025), round(end[1]/.025), round(end[2]/.10))
                    if key not in candidates or score > candidates[key][0]:
                        candidates[key] = item
                    if progress > minimum_progress and depth > 0:
                        solutions.append(item)
            if not candidates:
                break
            beam = sorted(candidates.values(), key=lambda item: item[0], reverse=True)[:beam_width]
            # Reobserve after the shortest useful escape; a longer route is
            # not worth exhausting the freshness budget of its sensor proof.
            if solutions:
                break
        if not solutions:
            return None
        self.recovery_path = max(solutions, key=lambda item: item[0])[2]
        return self.recovery_path[0][:2]

    def extended_recovery_path(self, yaw_error):
        self.recovery_path=None
        self._lookahead_recovery(yaw_error,.06,self.clearance(np.zeros(3)),
            depth_limit=12,beam_width=48,minimum_progress=.12)
        return self.recovery_path


class ObservedFootprintGuard:
    """ROS adapter shared by candidate selection and the final velocity gate."""
    def __init__(self, node):
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from tf2_ros import Buffer, TransformListener
        self.node = node
        node.declare_parameter('guard_footprint_bounds_m', [-.245, .312, -.226, .226])
        node.declare_parameter('guard_static_free_history_sec', 2.)
        node.declare_parameter('guard_observed_local_costmap_enabled', False)
        node.declare_parameter('guard_observed_local_costmap_topic', '/guard/observed_costmap')
        self.bounds = node.get_parameter('guard_footprint_bounds_m').value
        self.free_history_age_sec=float(node.get_parameter('guard_static_free_history_sec').value)
        if not math.isfinite(self.free_history_age_sec) or not 2.<=self.free_history_age_sec<=30.:
            raise ValueError('Static free-space memory must be bounded to 2..30 seconds')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        self.grid = None
        self.local_grid = None
        self.local_grid_enabled = bool(node.get_parameter('guard_observed_local_costmap_enabled').value)
        self.recovery = None
        self.free_history = deque(maxlen=math.ceil(self.free_history_age_sec/.12)+2)
        self.static_history_ready = False
        self.phase_sub = node.create_subscription(String, '/manipulation_phase', self.on_phase, 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub = node.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.on_grid, qos)
        if self.local_grid_enabled:
            self.local_sub = node.create_subscription(OccupancyGrid,
                str(node.get_parameter('guard_observed_local_costmap_topic').value), self.on_local_grid, qos)

    def on_grid(self, msg):
        self.grid = (msg, self.node.get_clock().now().nanoseconds*1.e-9)

    def on_local_grid(self, msg):
        self.local_grid = (msg, self.node.get_clock().now().nanoseconds*1.e-9)

    def on_phase(self, msg):
        ready = msg.data.split(':', 1)[0] == 'COMPLETE'
        if ready != self.static_history_ready:
            # Never reuse free rays from before/during a moving door.
            self.free_history.clear()
        self.static_history_ready = ready

    def reset_recovery(self):
        self.recovery=None
        self.last_recovery_progress_time=None
        self.recovery_segment_start=None
        pending=getattr(self,'pending_recovery',None)
        if pending is not None:
            pending[0].cancel()
        self.pending_recovery=None
        self.next_extended_plan=0.

    def recovery_command(self, scene, yaw_error, now):
        if scene is None:
            # A missing sample stops motion, but must not undo the retreat
            # part of an escape by forgetting its remaining turn. The original
            # primitive deadline and current-scene checks still apply on resume.
            active=getattr(self,'recovery',None)
            if active is not None and not 0.<=now-active[1]<=5.:
                self.recovery=None
                self.last_recovery_rejection='expired_during_sensor_gap'
            return 0., 0., 'STALE_GEOMETRY'
        # Local escape primitives must not move when SLAM adjusts map->odom.
        # Map pose is still used for observed-grid lookup, not servo progress.
        odom=getattr(scene,'odom_pose',None)
        tracking_pose=scene.pose if odom is None else np.asarray(odom,float)
        if np.shape(tracking_pose)!=(3,) or not np.isfinite(tracking_pose).all():
            self.recovery=None
            return 0.,0.,'INVALID_TRACKING_POSE'
        tracking_frame='map' if odom is None else 'odom'
        pending=getattr(self,'pending_recovery',None)
        if pending is not None:
            future,origin,frame,started=pending
            if not 0.<=now-started<=15.:
                self.reset_recovery()
                self.next_extended_plan=now+5.
                return 0.,0.,'EXPIRED_ESCAPE_PLAN'
            if not future.done():
                # Long rollouts run off the ROS callback thread, with the base
                # stopped. Their old scan proposes a route, never a command.
                return 0.,0.,'PLANNING_OBSERVED_ESCAPE'
            self.pending_recovery=None
            displacement=np.linalg.norm(tracking_pose[:2]-origin[:2])
            rotation=abs(math.atan2(math.sin(tracking_pose[2]-origin[2]),
                                    math.cos(tracking_pose[2]-origin[2])))
            try:
                path=future.result()
            except (ValueError,RuntimeError):
                path=None
            self.next_extended_plan=now+2.
            if (path and frame==tracking_frame and 0.<=now-started<=15.
                    and displacement<=.04 and rotation<=.06):
                self._store_recovery(path,origin,frame,now)
            # Every accepted primitive is checked below against today's scan.
        active = getattr(self, 'recovery', None)
        if active is not None and getattr(self,'recovery_frame',tracking_frame)!=tracking_frame:
            self.recovery=None
            active=None
        if active is not None:
            path, began = active
            v, w, goal = path[0]
            distance = np.linalg.norm(tracking_pose[:2]-goal[:2])
            angle = math.atan2(math.sin(tracking_pose[2]-goal[2]), math.cos(tracking_pose[2]-goal[2]))
            # Arc endpoint tolerance stays inside the 5cm body reserve. The
            # next primitive is still checked from the actual measured pose.
            position_tolerance=.04 if abs(v)>1.e-8 and abs(w)>1.e-8 else .018
            if distance <= position_tolerance and abs(angle) <= .04:
                origin=getattr(self,'recovery_segment_start',None)
                if origin is not None and tracking_frame=='odom':
                    travel=float(np.linalg.norm(tracking_pose[:2]-origin[:2]))
                    turn=abs(math.atan2(math.sin(tracking_pose[2]-origin[2]),
                                       math.cos(tracking_pose[2]-origin[2])))
                    dt=now-began
                    if (0.<dt<=5. and (travel>=.03 or turn>=.10)
                            and travel<=.12*dt+.05 and turn<=.5*dt+.06):
                        self.last_recovery_progress_time=now
                path = path[1:]
                self.recovery = (path, now) if path else None
                self.recovery_segment_start=tracking_pose.copy() if path else None
                if path:
                    v, w, goal = path[0]
                    began = now
            # Keep a selected escape sequence until measured subgoals are
            # reached. Recheck live clearance; never execute a stale plan.
            if self.recovery is not None:
                # Keep the independent 1.5-second safety prediction inside
                # this primitive. A full-speed prediction near its end would
                # extend beyond the planned turn and discard a valid escape.
                c, s = math.cos(tracking_pose[2]), math.sin(tracking_pose[2])
                angle = math.atan2(math.sin(tracking_pose[2]-goal[2]), math.cos(tracking_pose[2]-goal[2]))
                delta = goal[:2]-tracking_pose[:2]
                along = float(delta @ np.array([c, s]))
                cross = float(delta @ np.array([-s, c]))
                missed_translation=(abs(w)<1.e-8 and abs(along)<=.018
                                    and (abs(cross)>.018 or abs(angle)>.04))
                remaining = (max(0., -angle/w) if abs(w) > 1.e-8
                             else max(0., along/v) if abs(v) > 1.e-8 else 0.)
                scale = min(1., remaining/1.5)
                if not missed_translation and 0. <= now-began <= 5. and scale > 0.:
                    if scene.safe(v*scale, w*scale):
                        return v*scale, w*scale, 'OBSERVED_RECOVERY_SUBGOAL'
                    # Never execute a blocked sweep. A transient noisy return
                    # may clear on a fresh scan within the unchanged deadline;
                    # immediately choosing forward here can undo an escape.
                    self.last_recovery_rejection=getattr(scene,'last_failure','blocked_live_sweep')
                    return 0.,0.,'OBSERVED_RECOVERY_BLOCKED'
                self.last_recovery_rejection=(
                    'missed_endpoint' if missed_translation else
                    'primitive_timeout' if now-began>5. else 'no_remaining_motion')
                self.recovery = None
        command = scene.choose(yaw_error)
        if scene.recovery_path:
            self._store_recovery(scene.recovery_path,tracking_pose,tracking_frame,now)
        # A short, positively scored sweep already has current sensor support.
        # Let it expose the next view instead of replacing it with a stopped
        # long search. Extended planning is only for an actually blocked step.
        elif (command[2] in ('NO_OBSERVED_SWEEP','NO_OBSERVED_PROGRESS')
                and now>=getattr(self,'next_extended_plan',0.)
                and hasattr(scene,'extended_recovery_path')):
            old_task=getattr(self,'recovery_task',None)
            if old_task is not None and not old_task.done():
                return 0.,0.,'DRAINING_OBSERVED_ESCAPE'
            executor=getattr(self,'recovery_executor',None)
            if executor is None:
                executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='observed_escape')
                self.recovery_executor=executor
            future=executor.submit(scene.extended_recovery_path,yaw_error)
            self.recovery_task=future
            self.pending_recovery=(future,tracking_pose.copy(),tracking_frame,now)
            return 0.,0.,'PLANNING_OBSERVED_ESCAPE'
        return command

    def _store_recovery(self,local_path,origin,frame,now):
        c,s=math.cos(origin[2]),math.sin(origin[2])
        path=[]
        for v,w,local in local_path:
            goal=np.asarray(local,float).copy()
            goal[:2]=goal[:2]@np.array([[c,s],[-s,c]])+origin[:2]
            goal[2]+=origin[2]
            path.append((v,w,goal))
        self.recovery=(path,now)
        self.last_recovery_rejection=None
        self.recovery_frame=frame
        self.recovery_segment_start=np.asarray(origin,float).copy()

    @staticmethod
    def planar(tf):
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

    def snapshot(self, scan, received):
        from rclpy.time import Time
        from rclpy.duration import Duration
        from tf2_ros import TransformException
        now = self.node.get_clock().now().nanoseconds*1.e-9
        if scan is None:
            return None
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec*1.e-9
        if not all(math.isfinite(a) and 0. <= a <= .75 for a in (now-stamp, now-received)):
            return None
        try:
            at = Time.from_msg(scan.header.stamp)
            sensor = self.planar(self.tf.lookup_transform('base_link', scan.header.frame_id, at,
                                 timeout=Duration(seconds=.015)))
            pose = self.planar(self.tf.lookup_transform('map', 'base_link', at,
                               timeout=Duration(seconds=.015)))
            grid = None
            if self.grid is not None:
                candidate, receipt = self.grid
                grid_stamp = candidate.header.stamp.sec+candidate.header.stamp.nanosec*1.e-9
                if (candidate.header.frame_id == 'map' and 0. <= now-receipt <= 3.
                        and 0. <= now-grid_stamp <= 3.):
                    grid = candidate
            odom_pose=None
            try:
                odom_pose=self.planar(self.tf.lookup_transform('odom','base_link',at,
                    timeout=Duration(seconds=.015)))
            except TransformException:
                pass
            local_grid=None
            if getattr(self,'local_grid_enabled',False) and getattr(self,'local_grid',None) is not None:
                candidate,receipt=self.local_grid
                grid_stamp=candidate.header.stamp.sec+candidate.header.stamp.nanosec*1.e-9
                if (candidate.header.frame_id=='odom' and odom_pose is not None
                        and 0.<=now-receipt<=1. and 0.<=now-grid_stamp<=1.):
                    local_grid=candidate
            scene=ObservedSweep(scan,sensor,self.bounds,pose,grid,
                free_history=list(self.free_history),odom_pose=odom_pose,measurement_stamp=stamp,
                free_history_age_sec=(self.free_history_age_sec if self.static_history_ready else 2.),
                local_grid=local_grid)
            if odom_pose is not None:
                if self.free_history and stamp<self.free_history[-1][0]:
                    self.free_history.clear()
                if not self.free_history or stamp-self.free_history[-1][0]>=.12:
                    self.free_history.append((stamp,odom_pose,sensor,scan))
            return scene
        except (ValueError, TransformException):
            return None
