"""Sensor-independent guards for bounded, feedback-driven contact motion."""

import math
import numpy as np


class SceneObservationPending(ValueError):
    """Motion remains stopped while a contradictory scan is reobserved."""


class ObservedSceneGate:
    def __init__(self, fail_on_repeated_unsafe=True):
        self.pending=None
        self.fail_on_repeated_unsafe=fail_on_repeated_unsafe

    def observe(self,stamp,now,wall_now,clear):
        if not fresh_sample(stamp,now,.25) or not math.isfinite(wall_now):
            raise ValueError('Scene recheck requires fresh nonfuture observations')
        if self.pending is None:
            if clear:
                return
            self.pending=dict(start=now,wall=wall_now,last=None,clear=[],unsafe=[])
        pending=self.pending
        if now-pending['start']>2. or wall_now-pending['wall']>30.:
            raise ValueError('Contact scene remains ambiguous after stopped reobservation')
        if stamp!=pending['last']:
            if pending['last'] is not None and stamp<pending['last']:
                raise ValueError('Contact scene recheck stamps moved backwards')
            pending['last']=stamp
            key='clear' if clear else 'unsafe'
            other='unsafe' if clear else 'clear'
            pending[key].append(stamp)
            pending[other].clear()
        if (self.fail_on_repeated_unsafe and len(pending['unsafe'])>=3
                and pending['unsafe'][-1]-pending['unsafe'][0]>=.16):
            raise ValueError('Repeated observed obstacle violates chassis clearance')
        if len(pending['clear'])>=3 and pending['clear'][-1]-pending['clear'][0]>=.16:
            self.pending=None
            return
        raise SceneObservationPending('Stopped for independent LiDAR clearance rechecks')


class ObservedContactProgress:
    """Bounded progress watchdog for opening and measured posture correction."""

    def __init__(self, now, angle, stall_limit=20., recovery_limit=75.):
        if not all(math.isfinite(v) for v in (now, angle, stall_limit, recovery_limit)):
            raise ValueError('Invalid contact progress baseline')
        if not 0. < stall_limit < recovery_limit:
            raise ValueError('Invalid contact progress time limits')
        self.best_angle = angle
        self.last_opening = self.last_progress = now
        self.stall_limit, self.recovery_limit = stall_limit, recovery_limit
        self.target = None
        self.best_error = None
        self.heading_target = None
        self.best_heading_error = None

    def observe(self, now, angle, actual, preferred,
                actual_heading=None, preferred_heading=None):
        actual, preferred = np.asarray(actual, float), np.asarray(preferred, float)
        if (actual.shape != (3,) or preferred.shape != (3,)
                or not np.all(np.isfinite((actual, preferred)))
                or not all(math.isfinite(v) for v in (now, angle))
                or now < self.last_progress):
            raise ValueError('Invalid contact progress observation')
        opened = angle > self.best_angle + .003
        if opened:
            self.best_angle = angle
            self.last_opening = self.last_progress = now
        corrected = False
        if self.target is not None:
            # Compare to the previous target. Changing the planner's target
            # alone cannot manufacture measured correction progress.
            error = float(np.linalg.norm(actual[:2]-self.target[:2]))
            if error < self.best_error - .002:
                self.best_error = error
                self.last_progress = now
                corrected = True
        if self.target is None or np.linalg.norm(preferred[:2]-self.target[:2]) > .005:
            self.target = preferred.copy()
            self.best_error = float(np.linalg.norm(actual[:2]-self.target[:2]))
        if actual_heading is not None or preferred_heading is not None:
            if (actual_heading is None or preferred_heading is None
                    or not all(math.isfinite(v) for v in (actual_heading, preferred_heading))):
                raise ValueError('Invalid contact heading progress observation')
            if self.heading_target is not None:
                error = abs(math.atan2(math.sin(actual_heading-self.heading_target),
                                       math.cos(actual_heading-self.heading_target)))
                if error < self.best_heading_error - .005:
                    self.best_heading_error = error
                    self.last_progress = now
                    corrected = True
            # Replanning alone cannot reset the stall timer. Only measured
            # movement toward the previous heading is correction progress.
            changed = (self.heading_target is None or abs(math.atan2(
                math.sin(preferred_heading-self.heading_target),
                math.cos(preferred_heading-self.heading_target))) > .01)
            if changed:
                self.heading_target = preferred_heading
                self.best_heading_error = abs(math.atan2(
                    math.sin(actual_heading-preferred_heading),
                    math.cos(actual_heading-preferred_heading)))
        if now-self.last_opening > self.recovery_limit:
            raise ValueError('Measured posture recovery did not resume door opening within its bounded budget')
        if now-self.last_progress > self.stall_limit:
            raise ValueError('Held contact has no measured opening or posture correction progress')
        return opened, corrected


def confirmed_angle_progress(samples, initial, now):
    """Median of distinct recent panel observations, never a one-frame peak."""
    if not all(math.isfinite(value) for value in (initial, now)):
        raise ValueError('Invalid observed progress baseline')
    distinct = {}
    for sample in samples:
        stamp, angle = sample['stamp'], sample['angle']
        if not all(math.isfinite(value) for value in (stamp, angle)):
            raise ValueError('Nonfinite panel progress observation')
        if fresh_sample(stamp, now, .5):
            distinct[stamp] = abs(angle-initial)
    recent = sorted(distinct.items())[-5:]
    if len(recent) < 3 or recent[-1][0]-recent[0][0] < .12:
        return None
    return float(np.median([value for _, value in recent]))


def temporal_scan_consensus(aligned_ranges, return_uncertainty=False, return_reserve=False):
    """Conservative range estimates from distinct ego-aligned observations.

    No spatial median is used: a repeatedly observed one-beam object survives.
    New large innovations and three consecutive closer returns are retained.
    Insufficient observations fall back to the latest raw measurement.
    """
    values = np.asarray(aligned_ranges, float)
    if return_reserve and not return_uncertainty:
        raise ValueError('Per-ray reserve requires uncertainty diagnostics')
    if values.ndim != 2 or not len(values):
        raise ValueError('Invalid aligned scan history')
    result = values[-1].copy()
    applied_reserve = np.zeros(len(result))
    counts = np.isfinite(values).sum(axis=0)
    supported = np.isfinite(result) & (counts >= 7)
    if not supported.any():
        if return_reserve:
            return result,0,None,applied_reserve
        return (result, 0, None) if return_uncertainty else (result, 0)
    selected = values[:, supported]
    median = np.nanmedian(selected, axis=0)
    sigma = np.maximum(.005, 1.4826*np.nanmedian(abs(selected-median), axis=0))
    stable_pool = sigma[sigma <= .25*median]
    pooled = max(.005, float(np.median(stable_pool))) if len(stable_pool) else .005
    # One short ray history cannot calibrate its noise scale reliably. Pool
    # stable rays without spatially averaging away a thin obstacle.
    sigma = np.maximum(sigma, pooled)
    stable = sigma <= .25*median
    uncertainty = 3.*math.sqrt(math.pi/2.)*sigma/np.sqrt(counts[supported])
    per_ray_reserve = uncertainty.copy()
    lower = median-uncertainty
    latest = selected[-1]
    innovation = latest < median-4.*sigma-.005
    recent = selected[-3:]
    sustained = (np.isfinite(recent).all(axis=0)
                 & (recent < lower).all(axis=0))
    lower[innovation] = np.minimum(lower[innovation], latest[innovation])
    lower[sustained] = np.minimum(lower[sustained], recent[:, sustained].max(axis=0))
    # Foreground/background switches are not Gaussian range noise. Never turn
    # an occlusion boundary into an invented negative-distance obstacle.
    lower[~stable] = latest[~stable]
    creditable = stable&~innovation&~sustained
    applied_reserve[np.flatnonzero(supported)[creditable]] = per_ray_reserve[creditable]
    result[supported] = lower
    modeled = int(stable.sum())
    confidence = math.sqrt(2.*math.log(2.*len(result)/.001))
    uncertainty = (confidence*math.sqrt(math.pi/2.)*pooled/
                   math.sqrt(float(counts[supported][stable].min()))) if modeled else None
    if return_reserve:
        return result,modeled,uncertainty,applied_reserve
    return (result, modeled, uncertainty) if return_uncertainty else (result, modeled)


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


def chassis_supplied_opening_speed(point, normal, speed, yaw_rate, requested=None):
    """Absorb excess chassis motion, never add independent normal arm push."""
    point, normal = np.asarray(point, float), np.asarray(normal, float)
    if (point.shape != (3,) or normal.shape != (3,)
            or not np.all(np.isfinite((point, normal)))
            or abs(normal[2]) > 1.e-6 or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not math.isfinite(speed) or abs(speed) > .06
            or not math.isfinite(yaw_rate) or abs(yaw_rate) > .08
            or (requested is not None and (not math.isfinite(requested)
                                          or not 0. <= requested <= .06))):
        raise ValueError('Invalid chassis-supplied opening velocity')
    velocity = np.array([speed-yaw_rate*point[1], yaw_rate*point[0], 0.])
    supplied = max(0., float(velocity@normal))
    return supplied if requested is None else min(supplied, requested)


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
            or abs(speed) > .06 or duration <= 0. or max_step <= 0.
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


def bounded_wrist_yaw_lead(rotation, yaw_rate, remaining_duration, maximum_tip_shift, tip_extent):
    """Small active-command orientation lead, never a measured motion substitute.

    Translation remains measured-only. The lead vanishes at the probe end and
    reserves tip clearance inside the original contact-step budget.
    """
    rotation=np.asarray(rotation,float)
    values=(yaw_rate,remaining_duration,maximum_tip_shift,tip_extent)
    if (rotation.shape!=(3,3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1.e-6)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1.e-6)
            or not all(math.isfinite(v) for v in values)
            or abs(yaw_rate)>.08 or remaining_duration<0.
            or not 0.<maximum_tip_shift<=.002 or tip_extent<=maximum_tip_shift):
        raise ValueError('Invalid bounded wrist yaw lead')
    limit=min(.03,maximum_tip_shift/tip_extent)
    lead=float(np.clip(yaw_rate*remaining_duration,-limit,limit))
    c,s=math.cos(lead),math.sin(lead)
    inverse=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
    return inverse@rotation,lead


def measured_contact_endpoint(initial_pose, current_pose, point, normal, requested_advance,
                              max_step):
    """Exact measured SE2 compensation; normal advance is chassis-supplied."""
    initial, current = np.asarray(initial_pose, float), np.asarray(current_pose, float)
    point, normal = np.asarray(point, float), np.asarray(normal, float)
    if (any(v.shape != (3,) for v in (initial, current, point, normal))
            or not np.all(np.isfinite((initial, current, point, normal)))
            or abs(normal[2]) > 1.e-6 or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not all(math.isfinite(v) for v in (requested_advance, max_step))
            or not 0. <= requested_advance <= max_step or max_step <= 0.):
        raise ValueError('Invalid measured contact endpoint')
    c, s = math.cos(initial[2]), math.sin(initial[2])
    translation = np.r_[np.array([[c, s], [-s, c]])@(current[:2]-initial[:2]), 0.]
    yaw = math.atan2(math.sin(current[2]-initial[2]), math.cos(current[2]-initial[2]))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    supplied = max(0., float((translation+(rotation-np.eye(3))@point)@normal))
    advance = min(requested_advance, supplied)
    endpoint = rotation.T@(point+normal*advance-translation)
    if np.linalg.norm(endpoint-point) > max_step:
        raise ValueError('Measured compensation exceeds bounded contact step')
    return endpoint, advance


def measured_tangential_relaxation(target, actual, normal, maximum_error, maximum_step,
                                  normal_tolerance):
    """Relax only in the panel plane; never add normal push or vertical press."""
    target, actual, normal = (np.asarray(v, float) for v in (target, actual, normal))
    if (any(v.shape != (3,) for v in (target, actual, normal))
            or not np.all(np.isfinite((target, actual, normal)))
            or abs(normal[2]) > 1.e-6 or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not all(math.isfinite(v) for v in (maximum_error, maximum_step, normal_tolerance))
            or not 0. < maximum_step <= maximum_error or normal_tolerance <= 0.):
        raise ValueError('Invalid measured tangential relaxation')
    residual = actual-target
    if (np.linalg.norm(residual) > maximum_error
            or abs(residual@normal) > normal_tolerance or abs(residual[2]) > normal_tolerance):
        raise ValueError('Loaded contact has excessive normal or vertical error')
    tangent = np.array([-normal[1], normal[0], 0.])
    shift = float(np.clip(residual@tangent, -maximum_step, maximum_step))*tangent
    return target+shift


def measured_preload_release(target, actual, normal, maximum_error, maximum_step,
                             tolerance, prepress_height, retained_roll, minimum_roll):
    """Release excess downward preload without advancing into the panel.

    This is position-error-based compliance, not a force measurement. The
    retained wrist roll and observed pre-press height bound unloading; loss of
    contact or an already excessive error remains a stop condition.
    """
    target, actual, normal = (np.asarray(v, float) for v in (target, actual, normal))
    values = (maximum_error, maximum_step, tolerance, prepress_height,
              retained_roll, minimum_roll)
    if (any(v.shape != (3,) for v in (target, actual, normal))
            or not np.all(np.isfinite((target, actual, normal)))
            or not all(math.isfinite(v) for v in values)
            or abs(normal[2]) > 1.e-6 or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not 0. < maximum_step <= tolerance <= maximum_error
            or not 0. < minimum_roll < retained_roll):
        raise ValueError('Invalid measured preload release')
    residual = actual-target
    if (np.linalg.norm(residual) > maximum_error
            or abs(residual@normal) > tolerance or abs(residual[2]) > tolerance):
        raise ValueError('Loaded contact already exceeds preload-release bounds')
    # Cartesian settling tolerance is not a minimum lever travel. Rotation
    # confirms depression; height only prevents unloading past its baseline.
    if actual[2] >= prepress_height:
        raise ValueError('No measured vertical lever depression remains')
    released = target.copy()
    released[2] += min(maximum_step, max(0., residual[2]-.5*tolerance))
    return released


def measured_chassis_rest(first, last, hand, displacement_tolerance, minimum_span=.15):
    first, last = np.asarray(first,float),np.asarray(last,float)
    hand = np.asarray(hand,float)
    if (first.shape!=(4,) or last.shape!=(4,) or hand.shape!=(3,)
            or not np.all(np.isfinite((first,last))) or not np.all(np.isfinite(hand))
            or not math.isfinite(displacement_tolerance) or displacement_tolerance<=0.
            or not math.isfinite(minimum_span) or minimum_span<=0.
            or last[0]<first[0]):
        raise ValueError('Invalid measured chassis rest samples')
    if last[0]-first[0]<minimum_span:
        return False
    yaw = math.atan2(math.sin(last[3]-first[3]),math.cos(last[3]-first[3]))
    radius = max(float(np.linalg.norm(hand[:2])),.1)
    motion = np.r_[last[1:3]-first[1:3],radius*yaw]
    return bool(np.linalg.norm(motion)<=displacement_tolerance)


def observed_probe_progress(initial_pose, current_pose, point, normal,
                            speed, yaw_rate, duration, max_step):
    """Arm interpolation follows measured chassis motion, not elapsed time.

    The projected progress is also capped by the normal displacement supplied
    by the measured chassis pose. A blocked base cannot become an arm push.
    """
    advance, residual, _ = observed_probe_motion(initial_pose, current_pose, point, normal,
                                                speed, yaw_rate, duration, max_step)
    return advance, residual


def observed_probe_motion(initial_pose, current_pose, point, normal,
                          speed, yaw_rate, duration, max_step):
    """Separate bounded chassis travel from its available normal opening supply.

    Translation acceleration can lag steering. Once either checked chassis
    component has reached its segment, do not keep turning merely because the
    chassis has not supplied the requested normal hand advance yet.
    """
    initial, current = np.asarray(initial_pose,float),np.asarray(current_pose,float)
    point, normal = np.asarray(point,float),np.asarray(normal,float)
    if (any(value.shape != (3,) for value in (initial,current,point,normal))
            or not np.all(np.isfinite((initial,current,point,normal)))
            or not all(math.isfinite(v) for v in (speed,yaw_rate,duration,max_step))
            or abs(speed)>.06 or abs(yaw_rate)>.08 or duration<=0. or max_step<=0.
            or abs(normal[2])>1.e-6 or abs(np.linalg.norm(normal)-1.)>1.e-6):
        raise ValueError('Invalid measured chassis progress')
    yaw=math.atan2(math.sin(current[2]-initial[2]),math.cos(current[2]-initial[2]))
    c,s=math.cos(initial[2]),math.sin(initial[2])
    translation=np.array([[c,s],[-s,c]])@(current[:2]-initial[:2])
    planned_yaw=yaw_rate*duration
    planned_translation=(np.array([speed*duration,0.]) if abs(yaw_rate)<1.e-8 else
        speed/yaw_rate*np.array([math.sin(planned_yaw),1.-math.cos(planned_yaw)]))
    radius=max(float(np.linalg.norm(point[:2])),.1)
    expected=np.r_[planned_translation,radius*planned_yaw]
    measured=np.r_[translation,radius*yaw]
    length_squared=float(expected@expected)
    if length_squared<1.e-12:
        if np.linalg.norm(measured)>max_step:
            raise ValueError('Uncommanded chassis displacement during held contact')
        return 0.,float(np.linalg.norm(measured)),0.
    progress=float(measured@expected/length_squared)
    if progress < -.1 or progress > 1.25:
        raise ValueError('Measured chassis progress opposes or exceeds bounded probe')
    residual=float(np.linalg.norm(measured-np.clip(progress,0.,1.)*expected))
    if residual>max_step:
        raise ValueError('Measured chassis motion leaves planned contact segment')
    chassis_progress = progress
    translation_squared = float(planned_translation@planned_translation)
    if math.sqrt(translation_squared) > .05*max_step:
        chassis_progress = max(chassis_progress,
                               float(translation@planned_translation/translation_squared))
    if abs(radius*planned_yaw) > .05*max_step:
        chassis_progress = max(chassis_progress, yaw/planned_yaw)
    rotate=lambda angle: np.array([[math.cos(angle),-math.sin(angle)],
                                  [math.sin(angle),math.cos(angle)]])
    planned_supply=float((planned_translation+(rotate(planned_yaw)-np.eye(2))@point[:2])@normal[:2])
    measured_supply=float((translation+(rotate(yaw)-np.eye(2))@point[:2])@normal[:2])
    if planned_supply>1.e-6:
        progress=min(progress,max(0.,measured_supply)/planned_supply)
    return float(np.clip(progress,0.,1.)),residual,float(np.clip(chassis_progress,0.,1.))


def bounded_probe_pulse_scale(progress, duration, pulse_duration, feedback_age, sample_period):
    """Leave enough measured travel for delayed odometry and the stop command."""
    if (not all(math.isfinite(v) for v in
                (progress,duration,pulse_duration,feedback_age,sample_period))
            or not 0.<=progress<=1. or duration<=0. or pulse_duration<=0.
            or not 0.<=feedback_age<=.15 or not 0.<sample_period<=.15):
        raise ValueError('Invalid measured contact pulse timing')
    remaining=1.-progress
    # Two observed sample intervals cover receiving the move and stop commands.
    horizon=pulse_duration+feedback_age+2.*sample_period
    return min(remaining,remaining*duration/horizon)


def _nearest_velocity(requested, rows, bounds, hand_x, metric=None, geometry_cache=None):
    """Exact weighted projection onto a two-variable convex polygon."""
    rows, bounds = np.asarray(rows, float), np.asarray(bounds, float)
    metric = np.diag([1., hand_x**2]) if metric is None else np.asarray(metric, float)
    inverse_metric = np.linalg.inv(metric)
    requested = np.asarray(requested, float)
    if np.all(rows@requested <= bounds+1.e-9):
        return 0., requested.copy()
    projected = rows@inverse_metric
    denominator = np.einsum('ij,ij->i',projected,rows)
    valid = denominator > 1.e-12
    projections = requested+((bounds[valid]-rows[valid]@requested)/denominator[valid])[:,None]*projected[valid]
    key = (rows.shape, rows.tobytes(), bounds.tobytes()) if geometry_cache is not None else None
    vertices = geometry_cache.get(key) if geometry_cache is not None else None
    if vertices is None:
        first, second = np.triu_indices(len(rows), 1)
        a, b = rows[first], rows[second]
        determinant = a[:,0]*b[:,1]-a[:,1]*b[:,0]
        valid = abs(determinant) > 1.e-10
        a,b,determinant = a[valid],b[valid],determinant[valid]
        c,d = bounds[first[valid]],bounds[second[valid]]
        vertices = np.column_stack(((c*b[:,1]-a[:,1]*d)/determinant,
                                    (a[:,0]*d-c*b[:,0])/determinant))
        if geometry_cache is not None:
            # Vertices depend on constraints, not the requested velocity or
            # metric. Cache only feasible ones with the original tolerance.
            batches = [batch[np.all(batch@rows.T <= bounds[None,:]+1.e-9, axis=1)]
                       for batch in np.array_split(vertices, max(1, math.ceil(len(vertices)/128)))]
            vertices = np.vstack(batches)
            geometry_cache[key] = vertices
    candidates = np.vstack((projections,vertices))
    delta = candidates-requested
    costs = np.einsum('ij,ij->i',delta@metric,delta)
    order = np.argsort(costs,kind='stable')
    # Same exact candidates and feasibility tolerance as scalar enumeration.
    # Check the cheapest batches first, without a large N-squared-by-N array.
    for start in range(0,len(order),128):
        indexes = order[start:start+128]
        feasible = np.all(candidates[indexes]@rows.T <= bounds[None,:]+1.e-9,axis=1)
        if feasible.any():
            index = indexes[np.flatnonzero(feasible)[0]]
            return float(costs[index]), candidates[index].copy()
    return None


def observed_panel_docking_lanes(anchor, normal, segment, body_outline, edge_margin):
    """Centre the chassis within a currently observed panel, not a world aperture."""
    anchor, normal, segment, outline = (np.asarray(v,float) for v in
                                       (anchor,normal,segment,body_outline))
    if (anchor.shape!=(2,) or normal.shape!=(2,) or segment.shape!=(2,2)
            or outline.ndim!=2 or outline.shape[1]!=2
            or not all(np.all(np.isfinite(v)) for v in (anchor,normal,segment,outline))
            or abs(np.linalg.norm(normal)-1.)>1.e-6
            or not math.isfinite(edge_margin) or edge_margin<=0.):
        raise ValueError('Invalid observed panel docking support')
    tangent=np.array([-normal[1],normal[0]])
    ends=(segment-anchor)@tangent
    low,high=float(min(ends)),float(max(ends))
    if high-low < np.ptp(outline[:,1])+2.*edge_margin:
        raise ValueError('Observed panel is too narrow for a centred chassis stance')
    if not low-edge_margin<=0.<=high+edge_margin:
        raise ValueError('Observed grasp does not belong to this panel support')
    lower=float(outline[:,1].max()+edge_margin-high)
    upper=float(outline[:,1].min()-edge_margin-low)
    centre=float((outline[:,1].max()+outline[:,1].min()-low-high)/2.)
    return sorted(set([centre,*np.linspace(lower,upper,17)]),key=lambda v:abs(v-centre))


def contact_posture_command(point, normal, preferred, body_outline, margin=.03,
                            opening_speed=.012, gain=.25, door_yaw_rate=0.,
                            limit_posture_drift=False, scene_constraints=None,
                            allow_posture_recovery=False, arm_constraints=None,
                            observed_curvature=None, opening_priority_weight=.2,
                            heading_priority_weight=0., heading_tolerance_rad=None,
                            measured_heading_reference=False,
                            panel_anchor=None, panel_normal=None,
                            nominal_velocity=None, nominal_velocity_weight=20.,
                            projection_cache=None, prefer_nominal_recovery=False):
    """Keep a reachable hand posture and calibrated body clear of the panel.

    Solve x_dot=u*n_x-v+w*y and y_dot=u*n_y-w*x. Door radius and hinge
    coordinates are deliberately absent. A separate contact audit is required
    for fixed jambs/strike and arm links; this is only a panel-plane envelope.
    """
    point, normal, preferred = (np.asarray(v, float) for v in (point, normal, preferred))
    outline = np.asarray(body_outline, float)
    if projection_cache is not None and not isinstance(projection_cache, dict):
        raise ValueError('Invalid contact projection cache')
    if (point.shape != (3,) or normal.shape != (3,) or preferred.shape != (3,)
            or outline.ndim != 2 or outline.shape[1] != 2
            or not np.all(np.isfinite((point,normal,preferred)))
            or not np.all(np.isfinite(outline)) or point[0] <= .1
            or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not all(math.isfinite(v) and v > 0. for v in (margin,gain))
            or not math.isfinite(opening_speed) or opening_speed < 0.
            or not math.isfinite(opening_priority_weight) or opening_priority_weight <= 0.
            or not math.isfinite(heading_priority_weight) or heading_priority_weight < 0.
            or (heading_tolerance_rad is not None and
                (observed_curvature is None or not math.isfinite(heading_tolerance_rad)
                 or not 0.<heading_tolerance_rad<=.3))
            or (observed_curvature is not None and not math.isfinite(observed_curvature))
            or not math.isfinite(door_yaw_rate) or abs(door_yaw_rate) > .2):
        raise ValueError('Invalid hand/body posture feedback')
    target = preferred.copy()
    relative = gain*(target-point)
    nominal_request = None if nominal_velocity is None else np.asarray(nominal_velocity,float)
    if nominal_request is not None and (nominal_request.shape!=(2,)
            or not np.all(np.isfinite(nominal_request))
            or not -.04<=nominal_request[0]<=.06 or abs(nominal_request[1])>.08
            or not math.isfinite(nominal_velocity_weight) or nominal_velocity_weight<=0.):
        raise ValueError('Invalid contact lookahead nominal velocity')
    if prefer_nominal_recovery and (nominal_request is None or nominal_request[0]>=0.
                                   or not allow_posture_recovery):
        raise ValueError('Planned recovery needs a bounded reverse contact reference')
    if (panel_anchor is None) != (panel_normal is None):
        raise ValueError('Measured panel anchor and normal must be supplied together')
    anchor = point if panel_anchor is None else np.asarray(panel_anchor, float)
    plane_normal = normal if panel_normal is None else np.asarray(panel_normal, float)
    if (anchor.shape != (3,) or plane_normal.shape != (3,)
            or not np.all(np.isfinite((anchor, plane_normal)))
            or abs(np.linalg.norm(plane_normal)-1.) > 1.e-6
            or abs(plane_normal[2]) > 1.e-6
            or float(normal@plane_normal) < .95
            or np.linalg.norm((anchor-point)-((anchor-point)@plane_normal)*plane_normal) > 1.e-6):
        raise ValueError('Invalid independently measured panel barrier')
    gaps = (anchor[:2]-outline)@plane_normal[:2]
    tangent = np.array([-plane_normal[1], plane_normal[0]])
    derivatives = (anchor[:2]-outline)@tangent
    supplied = np.array([normal[0], normal[1]*point[0]-normal[0]*point[1]])
    corners = outline[:,1]*plane_normal[0]-outline[:,0]*plane_normal[1]
    gap_rows = np.column_stack((np.full(len(outline),plane_normal[0]), -corners))
    recovery = door_yaw_rate*derivatives+gain*(gaps-margin)
    # Actual opening u=min(request, max(0, chassis supply)) has three affine
    # regions. Enforce every body-corner barrier using THAT u, not an opening
    # speed the chassis cannot supply. Steering and drive must be projected
    # together; limiting drive alone cannot preserve clearance in region 2.
    opening_rows = (np.zeros(2), supplied, np.zeros(2))
    opening_offsets = (opening_speed, 0., 0.)
    region_rows = ((-supplied,), (supplied, -supplied), (supplied,))
    region_bounds = ((-opening_speed,), (opening_speed, 0.), (0.,))
    # The protruding grasp follows the hand tangent, while the measured panel
    # has its own plane. They need not be parallel or pass through the TCP.
    factor = (np.full(len(outline), float(normal@plane_normal)) if observed_curvature is None else
              float(normal@plane_normal)+observed_curvature*derivatives)
    # A held panel rotates in proportion to this step's supplied travel. An
    # old nonzero rate must not predict continued opening during zero travel.
    regions = tuple((np.vstack((base_rows, gap_rows-np.outer(factor, opening_row))),
                     np.r_[base_bounds, factor*offset+recovery])
                    for base_rows, base_bounds, opening_row, offset in
                    zip(region_rows, region_bounds, opening_rows, opening_offsets))
    if limit_posture_drift:
        error = point[:2]-target[:2]
        along = float(error@normal[:2])
        cross = error[0]*point[1]-error[1]*point[0]
        row = np.array([-error[0],cross])
        decrease = 0.
        posture_rows = (row, row+along*supplied, row)
        posture_bounds = (decrease-opening_speed*along, decrease, decrease)
        # A forward panel-clear command is not safe if it keeps stretching
        # the loaded arm. Do not increase distance from a reachable posture;
        # the nominal feedback still requests error reduction when feasible.
        regions = tuple((np.vstack((rows,posture_row)), np.r_[bounds,posture_bound])
                        for (rows,bounds),posture_row,posture_bound in
                        zip(regions,posture_rows,posture_bounds))
    if scene_constraints is not None:
        scene_rows, scene_bounds = (np.asarray(value,float) for value in scene_constraints)
        if (scene_rows.ndim != 2 or scene_rows.shape[1] not in (2,3)
                or scene_bounds.shape != (len(scene_rows),)
                or not np.all(np.isfinite(scene_rows)) or not np.all(np.isfinite(scene_bounds))):
            raise ValueError('Invalid observed scene constraints')
        if scene_rows.shape[1]==2:
            regions = tuple((np.vstack((rows,scene_rows)),np.r_[bounds,scene_bounds])
                            for rows,bounds in regions)
        else:
            regions = tuple((np.vstack((rows,scene_rows[:,:2]+np.outer(scene_rows[:,2],opening_row))),
                             np.r_[bounds,scene_bounds-scene_rows[:,2]*offset])
                            for (rows,bounds),opening_row,offset in
                            zip(regions,opening_rows,opening_offsets))
    box_rows = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    box_bounds = np.array([.06,.04,.08,.08])
    body_velocity = np.array([[1., -point[1]], [0., point[0]]])
    if arm_constraints is not None:
        arm_rows, arm_bounds = (np.asarray(value, float) for value in arm_constraints)
        if (arm_rows.ndim != 2 or arm_rows.shape[1] != 3
                or arm_bounds.shape != (len(arm_rows),)
                or not np.all(np.isfinite(arm_rows)) or not np.all(np.isfinite(arm_bounds))):
            raise ValueError('Invalid measured arm velocity constraints')

    def project_region(index, rows, bounds, forward_only=False):
        opening_row = supplied if index == 1 else np.zeros(2)
        opening_offset = opening_speed if index == 0 else 0.
        if heading_tolerance_rad is not None:
            sine=observed_curvature*target[0]
            if abs(sine)>=1.:
                raise ValueError('Preferred contact heading is outside observed articulation reach')
            heading_error=math.atan2(math.sin(math.atan2(normal[1],normal[0])-math.asin(sine)),
                                     math.cos(math.atan2(normal[1],normal[0])-math.asin(sine)))
            heading_row=observed_curvature*opening_row-np.array([0.,1.])
            heading_bias=observed_curvature*opening_offset
            rows=np.vstack((rows,heading_row,-heading_row))
            bounds=np.r_[bounds,
                gain*(heading_tolerance_rad-heading_error)-heading_bias,
                gain*(heading_tolerance_rad+heading_error)+heading_bias]
        if arm_constraints is not None:
            rows = np.vstack((rows, arm_rows[:, :2]+np.outer(arm_rows[:, 2], opening_row)))
            bounds = np.r_[bounds, arm_bounds-arm_rows[:, 2]*opening_offset]
        motion = np.outer(normal[:2], opening_row)-body_velocity
        error = opening_offset*normal[:2]-relative[:2]
        opening_error = opening_offset-opening_speed
        metric = motion.T@motion+opening_priority_weight*np.outer(opening_row, opening_row)
        rhs = -motion.T@error-opening_priority_weight*opening_error*opening_row
        heading_row, heading_error = np.zeros(2), 0.
        if observed_curvature is not None and heading_priority_weight:
            if abs(observed_curvature*target[0]) >= 1.:
                raise ValueError('Preferred contact heading is outside observed articulation reach')
            # During reconfiguration the hand has not attained the future
            # reach yet. Steering for that future reach can fold the actual
            # arm while opposing its Cartesian recovery.
            heading_x = min(point[0], target[0]) if measured_heading_reference else target[0]
            sine = observed_curvature*heading_x
            if abs(sine) >= 1.:
                raise ValueError('Preferred contact heading is outside observed articulation reach')
            desired_heading = math.asin(sine)
            heading_error = (observed_curvature*opening_offset
                             +gain*(math.atan2(normal[1],normal[0])-desired_heading))
            heading_row = observed_curvature*opening_row-np.array([0.,1.])
            metric += heading_priority_weight*np.outer(heading_row,heading_row)
            rhs -= heading_priority_weight*heading_error*heading_row
        tracking_metric = np.diag([1., point[0]**2])*nominal_velocity_weight
        if nominal_request is not None:
            metric += tracking_metric
            rhs += tracking_metric@nominal_request
        nominal = np.linalg.solve(metric, rhs)
        extra_rows = [[-1., 0.]] if forward_only else np.empty((0, 2))
        extra_bounds = [0.] if forward_only else []
        solution = _nearest_velocity(nominal,
            np.vstack((box_rows, extra_rows, rows)), np.r_[box_bounds, extra_bounds, bounds],
            point[0], metric=metric, geometry_cache=projection_cache)
        if solution is None:
            return None
        _, candidate = solution
        actual_error = motion@candidate+error
        actual_opening = opening_offset+opening_row@candidate
        heading_residual = heading_row@candidate+heading_error
        tracking_error = 0. if nominal_request is None else float(
            (candidate-nominal_request)@tracking_metric@(candidate-nominal_request))
        return (float(actual_error@actual_error+opening_priority_weight*(actual_opening-opening_speed)**2
                      +heading_priority_weight*heading_residual**2+tracking_error), candidate)

    # Keep forward-only opening whenever feasible. Reverse is a bounded
    # measured clearance recovery, never a timed default retreat.
    choices = []
    posture_setup=measured_heading_reference and heading_tolerance_rad is None
    # A differential-drive stance change may have zero door advance. Include
    # that forward-only region before considering a reverse recovery.
    for index, (rows, bounds) in enumerate(regions if posture_setup else regions[:2]):
        solution = project_region(index, rows, bounds, forward_only=True)
        if solution is not None:
            choices.append(solution)
    forward_opening = (float(np.clip(supplied@min(choices, key=lambda item: item[0])[1],
                                    0., opening_speed)) if choices else 0.)
    if not choices or (allow_posture_recovery and
                       ((forward_opening < .002 and not posture_setup) or prefer_nominal_recovery)):
        recovery_choices = []
        for index, (rows, bounds) in enumerate(regions):
            solution = project_region(index, rows, bounds)
            if solution is not None:
                recovery_choices.append(solution)
        # An always-feasible zero-velocity command must not prohibit a measured
        # reverse posture correction. The same panel/scene barriers still apply.
        if recovery_choices:
            choices = recovery_choices
    if not choices:
        raise ValueError('No panel-clear contact motion within calibrated velocity bounds')
    _, command = min(choices, key=lambda choice: choice[0])
    command = np.clip(command, [-.04, -.08], [.06, .08])
    command[np.abs(command) < 1.e-12] = 0.
    return float(command[0]), float(command[1]), target, float(min(gaps))


def contact_scene_planning_margin(range_uncertainty, pose_tolerance, accounted_clearance=None):
    """Count uncertainty once, including reserve in lower-bound scan points."""
    uncertainty = 0. if range_uncertainty is None else float(range_uncertainty)
    if (not math.isfinite(uncertainty) or uncertainty < 0.
            or not math.isfinite(pose_tolerance) or pose_tolerance <= 0.):
        raise ValueError('Invalid observed contact scene reserve')
    if accounted_clearance is None:
        return .03 + max(uncertainty, pose_tolerance)
    accounted = np.asarray(accounted_clearance, float)
    if (accounted.ndim != 1 or not np.all(np.isfinite(accounted))
            or np.any(accounted < 0.)):
        raise ValueError('Invalid accounted scan clearance reserve')
    return .03 + np.maximum(pose_tolerance, uncertainty-accounted)


def scanned_body_constraints(points, body_outline, margin=.03, gain=.25,
                             point_opening_motion=None, motion_uncertainty=None):
    """Static-scene barriers from observed scan points, never a stored map.

    Unclassified returns remain stationary. A visually supported panel return
    may have a bounded velocity per unit of chassis-supplied opening travel.
    """
    points, outline = np.asarray(points,float),np.asarray(body_outline,float)
    margin = np.asarray(margin, float)
    if (points.ndim != 2 or points.shape[1] != 2 or outline.ndim != 2
            or outline.shape[1] != 2 or len(points) < 3
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(outline))
            or (margin.ndim != 0 and margin.shape != (len(points),))
            or not np.all(np.isfinite(margin)) or np.any(margin <= 0.)
            or not math.isfinite(gain) or gain <= 0.):
        raise ValueError('Invalid observed chassis obstacle points')
    delta = points-np.clip(points,outline.min(axis=0),outline.max(axis=0))
    gaps = np.linalg.norm(delta,axis=1)
    if np.min(gaps) <= 0.:
        raise ValueError('Observed obstacle enters the calibrated chassis envelope')
    normal = delta/gaps[:,None]
    rows = np.column_stack((normal[:,0],-normal[:,0]*points[:,1]+normal[:,1]*points[:,0]))
    bounds = gain*(gaps-margin)
    if point_opening_motion is not None:
        motion=np.asarray(point_opening_motion,float)
        uncertainty=np.asarray(motion_uncertainty,float)
        if (motion.shape!=points.shape or uncertainty.shape!=(len(points),)
                or not np.all(np.isfinite(motion)) or not np.all(np.isfinite(uncertainty))
                or np.any(uncertainty<0.)):
            raise ValueError('Invalid observed panel return velocity')
        # Use the adverse end of the measured velocity uncertainty interval.
        rows=np.column_stack((rows,-np.einsum('ij,ij->i',normal,motion)+uncertainty))
    # Distant points cannot constrain any bounded velocity; excluding them
    # reduces the exact polygon solver's pairwise intersection cost.
    worst = np.abs(rows[:,0])*.06+np.abs(rows[:,1])*.08
    if rows.shape[1]==3:
        worst+=np.abs(rows[:,2])*.06
    active = bounds < worst+1.e-9
    return rows[active],bounds[active],float(np.min(gaps))


def observed_panel_return_motion(points, nominal, segment, panel_normal,
                                  hand, hand_normal, curvature,
                                  distance_tolerance=.03, edge_margin=.05,
                                  curvature_uncertainty=0.):
    """Bounded RGB-D segment identifies only interior panel scan returns.

    The rotation center is inferred from measured TCP travel and panel angles.
    Wall, jamb, unobserved edges and uncertain rays remain fixed obstacles.
    """
    points,nominal,segment=(np.asarray(v,float) for v in (points,nominal,segment))
    panel_normal,hand,hand_normal=(np.asarray(v,float) for v in (panel_normal,hand,hand_normal))
    if (points.ndim!=2 or points.shape[1]!=2 or nominal.shape!=points.shape
            or segment.shape!=(2,2) or panel_normal.shape!=(2,)
            or hand.shape!=(3,) or hand_normal.shape!=(3,)
            or not all(np.all(np.isfinite(v)) for v in
                       (points,nominal,segment,panel_normal,hand,hand_normal))
            or not all(math.isfinite(v) for v in (curvature,distance_tolerance,edge_margin,curvature_uncertainty))
            or abs(curvature)<1.e-6 or distance_tolerance<=0. or edge_margin<=0.
            or not 0.<=curvature_uncertainty<abs(curvature)
            or abs(np.linalg.norm(panel_normal)-1.)>1.e-6
            or abs(np.linalg.norm(hand_normal)-1.)>1.e-6 or abs(hand_normal[2])>1.e-6):
        raise ValueError('Invalid observed moving-panel support')
    axis=segment[1]-segment[0]
    length=float(np.linalg.norm(axis))
    if length<=2.*edge_margin:
        return np.zeros_like(points),np.zeros(len(points)),np.zeros(len(points),bool)
    axis/=length
    along=(nominal-segment[0])@axis
    distance=abs((nominal-segment[0])@panel_normal)
    selected=(along>=edge_margin)&(along<=length-edge_margin)&(distance<=distance_tolerance)
    hinge=hand[:2]+np.array([-hand_normal[1],hand_normal[0]])/curvature
    radial=nominal-hinge
    motion=np.column_stack((-radial[:,1],radial[:,0]))*curvature
    motion[~selected]=0.
    uncertainty=(abs(curvature)*np.linalg.norm(nominal-points,axis=1)
                 +curvature_uncertainty*np.linalg.norm(radial,axis=1))
    uncertainty[~selected]=0.
    return motion,uncertainty,selected


def released_grip_clearance(front_extent, contact_offset, measured_span, tolerance):
    """Withdraw the finger tip past the observed grasp span, plus pose margin."""
    values = (front_extent, contact_offset, measured_span, tolerance)
    if (not all(math.isfinite(x) for x in values) or not 0. < contact_offset <= front_extent
            or not .004 < measured_span <= .072 or tolerance <= 0.):
        raise ValueError('Invalid measured release clearance')
    return front_extent-contact_offset + measured_span/2. + tolerance


def measured_contact_preload(actual, normal, preload, maximum):
    """A non-accumulating compliant target anchored to current robot FK.

    This is bounded position preload, not a force measurement or force control.
    Verified grasp and separately observed door progress remain required.
    """
    actual, normal = np.asarray(actual,float), np.asarray(normal,float)
    if (actual.shape != (3,) or normal.shape != (3,)
            or not np.all(np.isfinite((actual,normal)))
            or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or not 0. <= preload <= maximum or not 0. < maximum <= .015):
        raise ValueError('Invalid measured contact preload')
    return actual + preload*normal


def robust_heading_rate(samples, maximum=.2, minimum_span=.25):
    """Median pairwise slope avoids differentiating one camera/FK switch.

    A sustained excessive rate is still rejected; the safety bound is not
    increased to make a noisy frame pass.
    """
    values=np.asarray(samples,float)
    if (values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values))
            or not all(math.isfinite(v) and v > 0. for v in (maximum,minimum_span))):
        raise ValueError('Invalid observed heading samples')
    if len(values) < 2:
        return None
    if np.any(np.diff(values[:,0]) <= 0.):
        raise ValueError('Observed heading stamps are not increasing')
    slopes=[(values[j,1]-values[i,1])/(values[j,0]-values[i,0])
            for i in range(len(values)) for j in range(i+1,len(values))
            if values[j,0]-values[i,0] >= minimum_span]
    if not slopes:
        return None
    rate=float(np.median(slopes))
    if abs(rate) > maximum:
        raise ValueError('Observed door heading rate is inconsistent')
    return rate


def active_heading_rate(samples, started, ended):
    """Use distinct sensor stamps inside a drive interval, excluding IK waits."""
    if not all(math.isfinite(value) for value in (started, ended)) or ended < started:
        raise ValueError('Invalid active motion interval')
    distinct = {}
    for stamp, angle in samples:
        if not all(math.isfinite(value) for value in (stamp, angle)):
            raise ValueError('Non-finite active heading observation')
        if started <= stamp <= ended:
            distinct[stamp] = angle
    values = sorted(distinct.items())
    if len(values) < 3:
        return None
    return robust_heading_rate(values, minimum_span=.16)


def measured_panel_margin(point, body_outline, uncertainty, floor=.03):
    """Convert measured panel angular uncertainty to a robot-size margin."""
    point, outline = np.asarray(point, float), np.asarray(body_outline, float)
    if (point.shape != (3,) or outline.ndim != 2 or outline.shape[1] != 2
            or not np.all(np.isfinite(point)) or not np.all(np.isfinite(outline))
            or not math.isfinite(uncertainty) or not 0. <= uncertainty <= math.radians(2.)
            or not math.isfinite(floor) or floor <= 0.):
        raise ValueError('Invalid panel clearance uncertainty')
    reach = float(np.max(np.linalg.norm(point[:2]-outline, axis=1)))
    return floor+reach*math.sin(uncertainty)


def qualified_panel_range(residual, range_uncertainty=None):
    """Keep legacy gates without a separately measured plane offset bound."""
    if not isinstance(residual,(int,float,np.number)) or not math.isfinite(residual):
        return False
    if range_uncertainty is None:
        return 0.<=residual<=.03
    return (isinstance(range_uncertainty,(int,float,np.number))
            and math.isfinite(range_uncertainty) and 0.<=range_uncertainty<=.03
            and 0.<=residual<=.06)


def qualified_panel_support(samples):
    """Keep only a qualified suffix; a bad latest exposure cannot be hidden."""
    accepted=[]
    for sample in sorted(samples,key=lambda value:value['stamp']):
        center=np.asarray(sample.get('center'),float)
        normal=np.asarray(sample.get('normal'),float)
        uncertainty=sample.get('uncertainty',float('nan'))
        residual=sample.get('residual',float('nan'))
        numeric=all(isinstance(value,(int,float,np.number)) for value in (uncertainty,residual))
        good=(center.shape==(3,) and normal.shape==(3,)
              and np.all(np.isfinite(center)) and np.all(np.isfinite(normal))
              and numeric and math.isfinite(uncertainty) and 0.<=uncertainty<=math.radians(2.)
              and qualified_panel_range(residual,sample.get('range_uncertainty'))
              and abs(np.linalg.norm(normal)-1.)<=1.e-6 and abs(normal[2])<=.15)
        if good:
            accepted.append(sample)
        else:
            accepted.clear()
    return accepted


def observed_panel_barrier_geometry(point, expected_normal, body_outline,
                                    centers, normals, stamps, uncertainties,
                                    residuals, now, calibration_tolerance, range_uncertainties=None):
    """A current depth/scan plane, independent of the protruding grasp point.

    Distinct observations must agree within sensor and bounded motion error.
    Range residual, angular uncertainty and acquisition delay remain explicit
    clearance reserves; robot jaw width is not a panel range measurement.
    """
    point, expected_normal, outline = (np.asarray(v, float) for v in
                                       (point, expected_normal, body_outline))
    centers, normals = (np.asarray(v, float) for v in (centers, normals))
    stamps, uncertainties, residuals = (np.asarray(v, float) for v in
                                         (stamps, uncertainties, residuals))
    count = len(stamps)
    if range_uncertainties is None:
        range_uncertainties=[None]*count
    if (point.shape != (3,) or expected_normal.shape != (3,)
            or outline.ndim != 2 or outline.shape[1] != 2
            or count < 3 or centers.shape != (count, 3) or normals.shape != (count, 3)
            or uncertainties.shape != (count,) or residuals.shape != (count,)
            or not all(np.all(np.isfinite(v)) for v in
                       (point, expected_normal, outline, centers, normals, stamps, uncertainties, residuals))
            or not all(math.isfinite(v) for v in (now, calibration_tolerance))
            or calibration_tolerance <= 0. or abs(np.linalg.norm(expected_normal)-1.) > 1.e-6
            or np.any(np.diff(stamps) <= 0.) or stamps[-1]-stamps[0] < .12
            or np.any(now-stamps < 0.) or np.any(now-stamps > .5) or now-stamps[-1] > .25
            or np.any(uncertainties < 0.) or np.any(uncertainties > math.radians(2.))
            or len(range_uncertainties)!=count
            or not all(qualified_panel_range(r,u) for r,u in zip(residuals,range_uncertainties))
            or np.any(abs(np.linalg.norm(normals, axis=1)-1.) > 1.e-6)
            or np.any(abs(normals[:, 2]) > .15)):
        raise ValueError('No consistent fresh measured panel barrier')
    normals = normals.copy()
    normals[:, 2] = 0.
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals[(normals@expected_normal) < 0.] *= -1.
    plane = normals[-1]
    if float(plane@expected_normal) < .95:
        raise ValueError('Panel barrier disagrees with the held articulation')
    offset = float((centers[-1]-point)@plane)
    if offset < -2.*calibration_tolerance:
        raise ValueError('Observed grasp is beyond the measured panel')
    anchor = point+offset*plane
    reach = float(np.max(np.linalg.norm(anchor[:2]-outline, axis=1)))
    maximum_plane_speed = .06+.2*reach
    for i in range(count-1):
        delay = stamps[-1]-stamps[i]
        disagreement = math.acos(float(np.clip(normals[i]@plane, -1., 1.)))
        if (disagreement > uncertainties[i]+uncertainties[-1]+.2*delay
                or abs(float((centers[i]-centers[-1])@plane)) >
                   calibration_tolerance+3.*max(residuals[i], residuals[-1])+maximum_plane_speed*delay):
            raise ValueError('Measured panel barrier is not temporally consistent')
    margin = (measured_panel_margin(anchor, outline, float(uncertainties.max()))
              +calibration_tolerance+3.*float(residuals.max())
              +maximum_plane_speed*(now-stamps[-1]))
    return anchor, plane, margin


def grasp_anchored_panel_margin(point, body_outline, angular_uncertainty,
                               finger_half_widths, calibration_tolerance):
    """Panel anchor comes from robot FK, not a LiDAR range measurement.

    Bound anchor ambiguity by the measured jaw envelope plus calibration
    tolerance. LiDAR range bounds remain in the separate scene barriers.
    """
    widths=np.asarray(finger_half_widths,float)
    if (widths.shape!=(2,) or not np.all(np.isfinite(widths))
            or not np.all((widths>.002)&(widths<=.036))
            or not math.isfinite(calibration_tolerance) or calibration_tolerance<=0.):
        raise ValueError('Missing bounded grasp anchor uncertainty')
    return (measured_panel_margin(point,body_outline,angular_uncertainty)
            +float(widths.max())+calibration_tolerance)


def measured_press_target(origin, actual, depth, maximum):
    """Press downward while following the measured lever's lateral travel.

    No lever pivot or radius is supplied. Reject excessive horizontal travel
    rather than pulling the grasp back to its old straight-line location.
    """
    origin,actual=np.asarray(origin,float),np.asarray(actual,float)
    if (origin.shape != (3,) or actual.shape != (3,)
            or not np.all(np.isfinite((origin,actual)))
            or not all(math.isfinite(v) for v in (depth,maximum))
            or not 0. <= depth <= maximum or maximum <= 0.
            or np.linalg.norm(actual[:2]-origin[:2]) > maximum):
        raise ValueError('Measured lever press travel exceeds safety envelope')
    goal=actual.copy()
    goal[2]=origin[2]-depth
    return goal


def predicted_contact_rotation(rotation, door_rate, base_rate, horizon=.5):
    """Short-horizon relative wrist prediction from observed motion rates."""
    if (not all(math.isfinite(v) for v in (door_rate,base_rate,horizon))
            or max(abs(door_rate),abs(base_rate)) > .2 or not 0. < horizon <= 1.):
        raise ValueError('Invalid observed relative rotation prediction')
    return measured_wrist_rotation(rotation,door_rate*horizon,base_rate*horizon)


def robust_contact_speed(samples, normal, maximum=.15, minimum_span=.25):
    """Measured directional travel rate, robust to one SLAM pose correction."""
    values=np.asarray(samples,float)
    normal=np.asarray(normal,float)
    if (values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values))
            or normal.shape != (3,) or not np.all(np.isfinite(normal))
            or abs(np.linalg.norm(normal)-1.) > 1.e-6):
        raise ValueError('Invalid measured contact travel samples')
    directional=np.column_stack((values[:,0],values[:,1:]@normal))
    try:
        return robust_heading_rate(directional,maximum,minimum_span)
    except ValueError as exc:
        raise ValueError('Measured contact travel rate is inconsistent') from exc


def observed_contact_curvature(samples, minimum_angle_span=.18, residual_limit=.008):
    """Fit rigid planar contact motion from sensor headings and measured TCPs.

    Rows are (stamp, observed panel yaw, stationary-frame TCP x, TCP y).
    No hinge, radius, world geometry or simulator state is an input. A short
    angular baseline cannot constrain radius reliably and returns no estimate.
    """
    values = np.asarray(samples, float)
    if (values.ndim != 2 or values.shape[1] != 4
            or not np.all(np.isfinite(values))
            or minimum_angle_span <= 0. or residual_limit <= 0.):
        raise ValueError('Invalid contact geometry observations')
    if len(values) < 12:
        return None
    if np.any(np.diff(values[:,0]) <= 0.):
        raise ValueError('Contact geometry stamps are not increasing')
    angles = np.unwrap(values[:,1])
    if np.ptp(angles) < minimum_angle_span:
        return None
    # Eliminate the unknown hinge by fitting all pairwise displacements.
    # Centering both sides also avoids catastrophic cancellation in map frames.
    c, s = np.cos(angles), np.sin(angles)
    rotations = np.stack((np.stack((c,-s),axis=1), np.stack((s,c),axis=1)),axis=1)
    design = (rotations-rotations.mean(axis=0)).reshape(-1,2)
    points = values[:,2:]
    centered = (points-points.mean(axis=0)).reshape(-1)
    radius_vector, _, _, _ = np.linalg.lstsq(design, centered, rcond=None)
    predicted = np.einsum('nij,j->ni',rotations,radius_vector)
    offset = np.median(points-predicted,axis=0)
    residuals = np.linalg.norm(points-predicted-offset,axis=1)
    keep = residuals <= max(residual_limit, 3.*float(np.median(residuals)))
    if keep.sum() < max(12, int(.8*len(values))) or np.ptp(angles[keep]) < minimum_angle_span:
        return None
    rotations, points = rotations[keep], points[keep]
    design = (rotations-rotations.mean(axis=0)).reshape(-1,2)
    centered = (points-points.mean(axis=0)).reshape(-1)
    radius_vector, _, _, _ = np.linalg.lstsq(design, centered, rcond=None)
    predicted = np.einsum('nij,j->ni',rotations,radius_vector)
    residuals = np.linalg.norm(points-predicted-(points-predicted).mean(axis=0),axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))
    radius = float(np.linalg.norm(radius_vector))
    if rms > residual_limit or np.max(residuals) > 3.*residual_limit or radius <= 10.*residual_limit:
        return None
    # Radius uncertainty from the angular baseline must remain smaller than
    # ten percent. A wrong pose frame or slipping grasp cannot become a path.
    sigma = max(rms, .002)/max(np.sin(np.ptp(angles[keep])/2.),1.e-6)
    if sigma/radius > .10:
        return None
    direction = np.sign(np.median(np.diff(angles[keep])))
    if direction == 0.:
        return None
    return dict(curvature_per_m=float(direction/radius), radius_m=radius,
                residual_rms_m=rms, radius_uncertainty_m=sigma,
                angular_span_rad=float(np.ptp(angles[keep])), samples=int(keep.sum()))


def conservative_contact_curvature(fit):
    """Reserve observed radius error when selecting a future arm/body stance."""
    curvature,radius,uncertainty=(float(fit[name]) for name in
        ('curvature_per_m','radius_m','radius_uncertainty_m'))
    if (not all(math.isfinite(v) for v in (curvature,radius,uncertainty))
            or radius<=0. or not 0.<=uncertainty<.1*radius
            or not math.isclose(abs(curvature)*radius,1.,rel_tol=1.e-6)):
        raise ValueError('Invalid contact curvature uncertainty')
    return curvature/(1.-uncertainty/radius)


def contact_steady_heading(curvature, forward_reach):
    """A future steady stance must be feasible, not clipped into a fake angle."""
    if (not all(math.isfinite(v) for v in (curvature,forward_reach))
            or forward_reach<=0. or abs(curvature*forward_reach)>=1.):
        raise ValueError('Preferred stance exceeds observed articulation reach')
    return math.asin(curvature*forward_reach)


def confirmed_plane_opening(samples, baseline, required, sim_now, wall_now):
    """At least three fresh geometric measurements above the uncertainty margin.

    RGB-D and visually initialized LiDAR planes are allowed. Encoder inference,
    controller setpoints and simulator joint values cannot confirm opening.
    """
    if not all(math.isfinite(v) for v in (baseline,required,sim_now,wall_now)) or required <= 0.:
        return None
    distinct={}
    for sample in samples:
        if sample.get('sensor') not in ('rgbd','lidar'):
            continue
        try:
            stamp,received,angle,uncertainty=(float(sample[key]) for key in
                ('stamp','received','angle','uncertainty'))
        except (KeyError,TypeError,ValueError):
            continue
        if (not all(math.isfinite(v) for v in (stamp,received,angle,uncertainty))
                or not 0. <= uncertainty <= math.radians(2.)
                or not 0. <= sim_now-stamp <= .75
                or not 0. <= wall_now-received <= 3.):
            continue
        distinct[stamp]=dict(stamp=stamp,received=received,angle=angle,
                             uncertainty=uncertainty,sensor=sample['sensor'])
    latest=max(distinct,default=float('-inf'))
    window=[distinct[stamp] for stamp in sorted(distinct) if latest-stamp <= .25]
    if (len(window) < 3 or window[-1]['stamp']-window[0]['stamp'] < .16
            or sim_now-window[-1]['stamp'] > .25):
        return None
    if any(abs(sample['angle']-baseline)-sample['uncertainty'] < required for sample in window):
        return None
    return window


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


def observed_press_increment(goal, actual, step, tracking_limit, pose_tolerance):
    """Keep room for the next measured loaded-servo response before pressing."""
    goal, actual = np.asarray(goal, float), np.asarray(actual, float)
    if (goal.shape != (3,) or actual.shape != (3,)
            or not np.all(np.isfinite((goal, actual)))
            or not all(math.isfinite(v) and v > 0.
                       for v in (step, tracking_limit, pose_tolerance))
            or pose_tolerance > tracking_limit):
        raise ValueError('Invalid observed press budget')
    error = goal-actual
    if np.linalg.norm(error) > pose_tolerance+1.e-9:
        raise ValueError('Loaded press exceeds calibrated pose tolerance')
    room = error[2]+math.sqrt(max(0., pose_tolerance**2-error[:2]@error[:2]))
    # Waiting is not contact success. Existing lever checks and deadlines still
    # decide whether a stationary loaded arm can continue safely.
    if room <= 1.e-5:
        return 0.
    return min(step, .5*room)


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
