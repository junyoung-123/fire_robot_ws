"""Sensor-independent guards for bounded, feedback-driven contact motion."""

import math
import numpy as np


def measured_wrist_rotation(reference, door_delta, base_yaw_delta, lever_delta=0.):
    """Follow a vertical hinge and panel-normal lever axis, in the base frame."""
    rotation = np.asarray(reference, dtype=float)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not all(math.isfinite(v) for v in (door_delta, base_yaw_delta, lever_delta))):
        raise ValueError('Invalid wrist rotation feedback')
    angle = door_delta - base_yaw_delta
    c, s = math.cos(angle), math.sin(angle)
    cl, sl = math.cos(lever_delta), math.sin(lever_delta)
    yaw = np.asarray(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    roll = np.asarray(((1., 0., 0.), (0., cl, -sl), (0., sl, cl)))
    return yaw @ roll @ rotation


def swept_contact_reference(reference, previous_tool, actual_tool, normal,
                            previous_angle, angle, max_displacement):
    """Infer tangential sweep from measured normal travel and hinge rotation.

    For a vertical hinge, dy/dx = tan(mid-angle). Radius and hinge/world
    position are not supplied. Measured lateral servo error is NOT integrated.
    The initial planar approach normal must be aligned with the closed panel.
    """
    points = np.asarray((reference, previous_tool, actual_tool, normal), dtype=float)
    if (points.shape != (4, 3) or not np.all(np.isfinite(points))
            or not all(math.isfinite(v) for v in (previous_angle, angle, max_displacement))
            or max_displacement <= 0. or max(abs(previous_angle), abs(angle)) > .6):
        raise ValueError('Invalid or excessive contact sweep feedback')
    axis = points[3].copy()
    axis[2] = 0.
    length = np.linalg.norm(axis)
    if length < .9:
        raise ValueError('Contact normal must be horizontal and normalized')
    axis /= length
    tangent = np.asarray((-axis[1], axis[0], 0.))
    delta = points[2] - points[1]
    if np.linalg.norm(delta) > max_displacement:
        raise ValueError('Measured tool jumped during contact sweep')
    advance = float(delta @ axis)
    result = points[0] + advance * axis + advance * math.tan((previous_angle+angle)/2.) * tangent
    return result


def projected_contact_increment(reference, previous_tool, actual_tool, tangent, max_displacement):
    """Project measured displacement onto the rotating door's local tangent.

    The chord of a circular handle path follows the midpoint panel normal.
    This does not divide by cos(angle), require a hinge coordinate, or infer
    radius from a small noisy motion. IK/tracking guards still bound the arm.
    """
    points = np.asarray((reference, previous_tool, actual_tool, tangent), dtype=float)
    if (points.shape != (4, 3) or not np.all(np.isfinite(points))
            or not math.isfinite(max_displacement) or max_displacement <= 0.):
        raise ValueError('Invalid contact projection feedback')
    axis = points[3].copy()
    if abs(axis[2]) > 1.e-6 or np.linalg.norm(axis) < .9:
        raise ValueError('Contact tangent must be horizontal and normalized')
    axis /= np.linalg.norm(axis)
    delta = points[2]-points[1]
    if np.linalg.norm(delta) > max_displacement:
        raise ValueError('Measured tool jumped during contact projection')
    return points[0] + float(delta@axis)*axis


def coordinated_contact_endpoint(point, normal, speed, yaw_rate, duration, max_step,
                                 opening_speed=None):
    """Compensate base-induced motion perpendicular to the opening direction.

    This is a bounded local velocity probe, not a preset hinge/radius path.
    The base supplies the forward component; the arm cancels sideways drag.
    """
    point, normal = np.asarray(point, float), np.asarray(normal, float)
    if (point.shape != (3,) or normal.shape != (3,)
            or not np.all(np.isfinite((point, normal)))
            or not all(math.isfinite(x) for x in (speed, yaw_rate, duration, max_step))
            or speed < 0. or duration <= 0. or max_step <= 0.
            or abs(normal[2]) > 1.e-6 or abs(np.linalg.norm(normal)-1.) > 1.e-6):
        raise ValueError('Invalid coordinated contact probe')
    velocity = np.array([speed-yaw_rate*point[1], yaw_rate*point[0], 0.])
    if opening_speed is not None and (not math.isfinite(opening_speed) or opening_speed < 0.):
        raise ValueError('Invalid observed contact opening speed')
    advance = (float(velocity@normal) if opening_speed is None else opening_speed)*duration
    if advance < -1.e-9:
        raise ValueError('Base probe would close the door')
    angle = yaw_rate*duration
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
    translation = (np.array([speed*duration,0.,0.]) if abs(yaw_rate) < 1.e-8 else
                   np.array([speed/yaw_rate*s, speed/yaw_rate*(1.-c), 0.]))
    endpoint = rotation.T@(point + normal*advance-translation)
    if max(np.linalg.norm(endpoint-point), abs(advance)) > max_step:
        raise ValueError('Coordinated probe exceeds bounded contact step')
    return endpoint, rotation.T


def contact_posture_command(point, normal, preferred, body_outline, margin=.03,
                            opening_speed=.012, gain=.25):
    """Keep a reachable hand posture and calibrated body clear of the panel.

    Solve x_dot=u*n_x-v+w*y and y_dot=u*n_y-w*x. Door radius and hinge
    coordinates are deliberately absent. A separate contact audit is required
    for fixed jambs/strike and arm links; this is only a panel-plane envelope.
    """
    point, normal, preferred = (np.asarray(v, float) for v in (point, normal, preferred))
    outline = np.asarray(body_outline, float)
    if (point.shape != (3,) or normal.shape != (3,) or preferred.shape != (3,)
            or outline.ndim != 2 or outline.shape[1] != 2
            or not np.all(np.isfinite((point,normal,preferred)))
            or not np.all(np.isfinite(outline)) or point[0] <= .1
            or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not all(math.isfinite(v) and v > 0. for v in (margin,opening_speed,gain))):
        raise ValueError('Invalid hand/body posture feedback')
    support = float(np.max(outline@normal[:2]))
    target = preferred.copy()
    if abs(normal[1]) > .1:
        bound = (support+margin-normal[0]*target[0])/normal[1]
        target[1] = min(target[1],bound) if normal[1] < 0. else max(target[1],bound)
    elif target[:2]@normal[:2] < support+margin:
        raise ValueError('No panel-clear preferred hand posture')
    relative = gain*(target-point)
    yaw_rate = float(np.clip((opening_speed*normal[1]-relative[1])/point[0], -.08, .08))
    speed = float(np.clip(opening_speed*normal[0]-relative[0]+yaw_rate*point[1], 0., .06))
    return speed, yaw_rate, target, float(point[:2]@normal[:2]-support)


def released_grip_clearance(front_extent, contact_offset, measured_span, tolerance):
    """Withdraw the finger tip past the observed grasp span, plus pose margin."""
    values = (front_extent, contact_offset, measured_span, tolerance)
    if (not all(math.isfinite(x) for x in values) or not 0. < contact_offset <= front_extent
            or not .004 < measured_span <= .072 or tolerance <= 0.):
        raise ValueError('Invalid measured release clearance')
    return front_extent-contact_offset + measured_span/2. + tolerance


class SimMotionDeadline:
    """Bound simulated motion separately from a stalled clock and wall cap."""

    def __init__(self, sim_now, wall_now, sim_limit, stall_limit=3.0, wall_limit=600.0):
        values = (sim_now, wall_now, sim_limit, stall_limit, wall_limit)
        if not all(math.isfinite(v) for v in values) or min(values[2:]) <= 0:
            raise ValueError('Invalid motion clock limits')
        self.started_sim = self.last_sim = sim_now
        self.started_wall = self.last_wall = self.progress_wall = wall_now
        self.sim_limit, self.stall_limit, self.wall_limit = values[2:]

    def failure(self, sim_now, wall_now):
        if not all(math.isfinite(v) for v in (sim_now, wall_now)):
            return 'Non-finite motion clock'
        if sim_now < self.last_sim or wall_now < self.last_wall:
            return 'Motion clock moved backwards'
        if sim_now > self.last_sim:
            self.progress_wall = wall_now
        self.last_sim, self.last_wall = sim_now, wall_now
        if sim_now - self.started_sim >= self.sim_limit:
            return 'Simulation motion deadline exceeded'
        if wall_now - self.progress_wall >= self.stall_limit:
            return 'Simulation clock stalled'
        if wall_now - self.started_wall >= self.wall_limit:
            return 'Absolute wall deadline exceeded'
        return None


def fresh_sample(stamp, now, max_age):
    return (stamp is not None and math.isfinite(stamp)
            and math.isfinite(now) and 0.0 <= now - stamp <= max_age)


def observed_clearance(points, minimum, maximum):
    """Safety floor plus measured spread, never a fixed approach distance.

    Repeated detections only estimate repeatability, not systematic depth bias.
    Reject excessive spread rather than silently clipping its safety margin.
    """
    values = np.asarray(points, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 3
            or not np.all(np.isfinite(values))):
        raise ValueError('At least three finite handle observations required')
    center = np.median(values, axis=0)
    spread = float(np.max(np.linalg.norm(values - center, axis=1)))
    clearance = minimum + 2.0 * spread
    if not 0.0 < minimum <= clearance <= maximum:
        raise ValueError('Handle uncertainty exceeds approach clearance budget')
    return center, clearance, spread


def bounded_press_step(goal, actual, step, tracking_limit):
    """Bound the next downward command, not only the previous target error."""
    goal, actual = np.asarray(goal, dtype=float), np.asarray(actual, dtype=float)
    if (goal.shape != (3,) or actual.shape != (3,)
            or not np.all(np.isfinite((goal, actual)))
            or not all(math.isfinite(v) and v > 0. for v in (step, tracking_limit))):
        raise ValueError('Invalid press tracking feedback')
    error = goal - actual
    if np.linalg.norm(error) > tracking_limit + 1.e-9:
        raise ValueError('Press tracking error: possible jam/contact loss')
    room = error[2] + math.sqrt(max(0., tracking_limit**2 - np.dot(error[:2], error[:2])))
    if room <= 1.e-5:
        raise ValueError('Press tracking error: no room for another bounded command')
    return min(step, room)


def next_press_depth(depth, step, limit, initial, current, sign, required):
    """Return (next depth, verified); travel limit is failure, not success."""
    values = (depth, step, limit, initial, current, sign, required)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Non-finite press feedback')
    if not (0 <= depth <= limit and step > 0 and required > 0
            and sign in (-1.0, 1.0)):
        raise ValueError('Invalid press safety settings')
    if sign * (current - initial) >= required:
        return depth, True
    if depth >= limit - 1.e-9:
        raise ValueError('Press travel limit reached without lever feedback')
    return min(limit, depth + step), False


def stable_lever_motion(samples, since, initial, sign, required, hold_sec=.08):
    """Confirm a duration, not a sample count (publisher rates may differ)."""
    if not samples:
        return False
    end = samples[-1][0]
    window = [s for s in samples if max(since, end - 2.0 * hold_sec) <= s[0] <= end]
    return (len(window) >= 3 and window[-1][0] - window[0][0] >= hold_sec
            and all(math.isfinite(t) and math.isfinite(v)
                    and sign * (v - initial) >= required for t, v in window))


def bounded_contact_target(observed, reference, tool_reference, actual,
                           max_shift, max_step):
    """Follow measured handle displacement, retaining the measured grip offset.

    All inputs must share a stationary frame. Never interpret a distant new
    detection as permission to jump the arm or extend its reach.
    """
    points = np.asarray((observed, reference, tool_reference, actual), dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError('Invalid observed contact coordinates')
    if not (math.isfinite(max_shift) and math.isfinite(max_step)
            and 0.0 < max_step <= max_shift):
        raise ValueError('Invalid contact tracking limits')
    displacement = points[0] - points[1]
    if np.linalg.norm(displacement) > max_shift:
        raise ValueError('Moving handle observation is inconsistent with locked contact')
    target = points[2] + displacement
    delta = target - points[3]
    distance = float(np.linalg.norm(delta))
    if distance > max_step:
        target = points[3] + delta * max_step / distance
    return target
