"""Check a short manual-approach segment in Nav2's inflated occupancy grid."""
import math


def retreat_progress_budget(anchor, lateral, odom_xy, elapsed, deadline):
    """Credit real inward progress only, bounded by a 300-second hard stop."""
    if (odom_xy is None or len(odom_xy) != 2
            or not all(math.isfinite(v) for v in (lateral, elapsed, deadline, *odom_xy))
            or elapsed < 0. or elapsed > deadline or elapsed >= 300.):
        return anchor, deadline
    sample = (abs(lateral), float(odom_xy[0]), float(odom_xy[1]), elapsed)
    if anchor is None:
        return sample, deadline
    gain = anchor[0]-sample[0]
    distance = math.hypot(sample[1]-anchor[1], sample[2]-anchor[2])
    dt = elapsed-anchor[3]
    # An inward SLAM correction earns no credit, but its old anchor must not
    # invalidate every subsequent genuine, independently measured wheel step.
    if dt>0. and gain>distance+.05:
        return sample,deadline
    # Map-only corrections, stationary turns and repeated oscillations cannot
    # renew the deadline. Wheel motion must support a new inward best position.
    if (dt > 0. and gain >= .08 and distance >= .08
            and gain <= distance+.05 and distance <= .12*dt+.05):
        return sample, max(deadline, min(300., elapsed+20.))
    return anchor, deadline


def remembered_return_heading(anchor, map_pose, odom_pose, waypoint, fallback_yaw):
    """Freeze a sensor-derived map return target in local odometry once."""
    if (len(map_pose)!=3 or len(odom_pose)!=3
            or not all(math.isfinite(v) for v in (*map_pose,*odom_pose,fallback_yaw))):
        raise ValueError('Invalid measured return pose')
    if anchor is None:
        turn=odom_pose[2]-map_pose[2]
        c,s=math.cos(turn),math.sin(turn)
        goal=None
        if waypoint is not None:
            if len(waypoint)<2 or not all(math.isfinite(v) for v in waypoint[:2]):
                raise ValueError('Invalid observed return waypoint')
            dx,dy=waypoint[0]-map_pose[0],waypoint[1]-map_pose[1]
            goal=(odom_pose[0]+c*dx-s*dy,odom_pose[1]+s*dx+c*dy)
        anchor=(goal,math.atan2(math.sin(fallback_yaw+turn),math.cos(fallback_yaw+turn)))
    goal,heading=anchor
    if goal is not None:
        dx,dy=goal[0]-odom_pose[0],goal[1]-odom_pose[1]
        if math.hypot(dx,dy)<=.18:
            anchor=(None,heading)
        else:
            heading=math.atan2(dy,dx)
    error=math.atan2(math.sin(heading-odom_pose[2]),math.cos(heading-odom_pose[2]))
    return anchor,error


def retreat_escape_budget(elapsed, deadline, now, confirmed_progress):
    """A wheel-confirmed safe escape subgoal may renew, never remove, the cap."""
    if (confirmed_progress is None
            or not all(math.isfinite(v) for v in (elapsed,deadline,now,confirmed_progress))
            or not 0.<=elapsed<=deadline or elapsed>=180.
            or not 0.<=now-confirmed_progress<=1.):
        return deadline
    return max(deadline,min(180.,elapsed+20.))


def retreat_scan_clearance(scan, now_sec, received_sec, timeout_sec=.75):
    """Fresh full scan clearance; no-return rays count as observed free space."""
    if scan is None or not scan.ranges:
        return None
    stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
    if any(not math.isfinite(age) or age < -1e-6 or age > timeout_sec
           for age in (now_sec-stamp, now_sec-received_sec)):
        return None
    step = float(scan.angle_increment)
    if step <= 0 or not math.isfinite(step):
        return None
    sectors = [[], [], []]
    for i, value in enumerate(scan.ranges):
        angle = math.atan2(math.sin(scan.angle_min+i*step), math.cos(scan.angle_min+i*step))
        indices = [2]
        if abs(angle) <= math.radians(35):
            indices.append(0)
        if abs(angle) >= math.radians(145):
            indices.append(1)
        if math.isinf(value) and value > 0:
            distance = float('inf')
        elif math.isfinite(value) and scan.range_min <= value <= scan.range_max:
            distance = float(value)
        else:
            distance = 0.0
        for index in indices:
            sectors[index].append(distance)
    if not all(sectors) or step*(len(scan.ranges)-1) < math.radians(350):
        return None
    return tuple(min(values) for values in sectors)


def observed_retreat_velocity(clearance, yaw_error, speed, angular_speed,
                              yaw_tolerance, turn_min, front_min=.48, rear_min=.40):
    """Clear the chassis before turning; reverse only when the measured escape requires it."""
    if clearance is None:
        return 0.0, 0.0, 'STALE_SCAN'
    front, rear, surround = clearance
    if surround < turn_min:
        preferred = 1 if math.cos(yaw_error) >= 0 else -1
        available = {1: front >= front_min, -1: rear >= rear_min}
        for direction in (preferred, -preferred):
            if available[direction]:
                return direction*min(abs(speed), .06), 0.0, 'CLEAR_BEFORE_TURN'
        return 0.0, 0.0, 'BLOCKED'
    if abs(yaw_error) > yaw_tolerance:
        return 0.0, math.copysign(abs(angular_speed), yaw_error), 'TURN_CLEAR'
    if front < front_min:
        return 0.0, 0.0, 'BLOCKED'
    return min(abs(speed), .12), 0.0, 'CENTERWARD'


def grid_segment_is_clear(grid, start, end, max_cost=98):
    if grid is None or not all(math.isfinite(v) for v in (*start, *end)):
        return False
    info=grid.info
    resolution=float(info.resolution)
    if (not math.isfinite(resolution) or resolution <= 0
            or info.width <= 0 or info.height <= 0
            or len(grid.data) != info.width*info.height):
        return False
    origin=info.origin.position
    q=info.origin.orientation
    if not all(math.isfinite(v) for v in (origin.x,origin.y,q.x,q.y,q.z,q.w)):
        return False
    if abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.) > .02:
        return False
    yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    c,s=math.cos(yaw),math.sin(yaw)
    steps=max(1,math.ceil(math.dist(start,end)/(resolution*.4)))
    if steps > 10000:
        return False
    for i in range(steps+1):
        x=start[0]+(end[0]-start[0])*i/steps-origin.x
        y=start[1]+(end[1]-start[1])*i/steps-origin.y
        gx=math.floor((c*x+s*y)/resolution)
        gy=math.floor((-s*x+c*y)/resolution)
        if not (0 <= gx < info.width and 0 <= gy < info.height):
            return False
        cost=grid.data[gy*info.width+gx]
        if not 0 <= cost <= max_cost:
            return False
    return True
