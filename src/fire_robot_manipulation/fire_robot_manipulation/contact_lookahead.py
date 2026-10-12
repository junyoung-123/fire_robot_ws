"""Robot-kinematic lookahead in a freshly observed contact scene.

Predictions select a nominal chassis velocity, not an opening certificate.
Fresh encoder/scene barriers and measured-motion execution remain mandatory.
"""
from dataclasses import dataclass, replace
import math
import numpy as np

from .piper_actual_kinematics import PIPER_JOINT_LIMITS
from .contact_feedback import contact_posture_command, scanned_body_constraints, contact_steady_heading


@dataclass
class ContactPlan:
    velocity: tuple
    predicted_opening_rad: float
    minimum_clearance_m: float
    minimum_condition: float
    evaluated: int
    projected_velocity: tuple = None
    execution_interval_sec: float = 1.
    arcs: tuple = ()
    arc_index: int = 0
    observed_curvature: float = None


def continue_contact_plan(plan,point,normal,curvature,curvature_uncertainty):
    """Continue only after the previous prediction matches real observations."""
    if not plan.arcs or not 0<=plan.arc_index<len(plan.arcs)-1:
        return None
    point,normal=np.asarray(point,float),np.asarray(normal,float)
    if (point.shape!=(3,) or normal.shape!=(3,) or not np.all(np.isfinite((point,normal)))
            or abs(np.linalg.norm(normal)-1.)>1.e-6
            or curvature_uncertainty is None or plan.observed_curvature is None
            or not all(math.isfinite(v) for v in (curvature,curvature_uncertainty))
            or curvature_uncertainty<0.
            or abs(curvature-plan.observed_curvature)>2.*curvature_uncertainty):
        return None
    previous=plan.arcs[plan.arc_index]
    if (np.linalg.norm(point-np.array(previous['tool_xyz']))>.015
            or np.linalg.norm(normal-np.array(previous['normal_xyz']))>.08):
        return None
    next_index=plan.arc_index+1
    arc=plan.arcs[next_index]
    return replace(plan,velocity=arc['velocity'],projected_velocity=arc['projected_velocity'],
                   arc_index=next_index)


def _rotate(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def contact_prediction_score(opened, hand_error, minimum_condition,
                             relative_heading=None, desired_heading=None,path_error=None):
    """Soft prediction ranking; physical constraints are checked separately."""
    values=(opened,hand_error,minimum_condition)
    if (not all(math.isfinite(value) for value in values)
            or min(values)<0. or (relative_heading is None)!=(desired_heading is None)):
        raise ValueError('Invalid contact prediction objective')
    reserve=max(0.,.28-minimum_condition)
    if path_error is not None and (not math.isfinite(path_error) or path_error<0.):
        raise ValueError('Invalid contact path tracking cost')
    path_cost=0. if path_error is None else 1.5*path_error
    if desired_heading is None:
        return opened-.8*hand_error-2.*reserve-path_cost
    if not all(math.isfinite(value) for value in (relative_heading,desired_heading)):
        raise ValueError('Invalid observed contact heading objective')
    error=relative_heading-desired_heading
    heading_error=abs(math.atan2(math.sin(error),math.cos(error)))
    return opened-1.5*hand_error-.5*heading_error-2.*reserve-path_cost


def plan_contact_motion(kinematics, joints, normal, curvature, scene_points,
                        moving_panel, scene_margin, panel_anchor, panel_normal,
                        panel_margin, preferred, opening_speed=.03,
                        depth=8, beam_width=6, interval=1.,
                        moving_uncertainty=None, controller_options=None):
    """Search bounded forward/recovery arcs without world coordinates or IDs."""
    q, normal, points, anchor, plane, preferred = (
        np.asarray(value, float) for value in
        (joints, normal, scene_points, panel_anchor, panel_normal, preferred))
    moving = np.asarray(moving_panel, bool)
    uncertainty = (np.zeros(len(points)) if moving_uncertainty is None else
                   np.asarray(moving_uncertainty, float))
    options = {} if controller_options is None else dict(controller_options)
    allowed = {'gain', 'opening_priority_weight', 'heading_priority_weight',
               'heading_tolerance_rad', 'measured_heading_reference'}
    if set(options)-allowed:
        raise ValueError('Invalid contact prediction controller options')
    margin = np.asarray(scene_margin, float)
    if margin.ndim == 0:
        margin = np.full(len(points), float(margin))
    if (q.shape != (6,) or normal.shape != (3,) or points.ndim != 2
            or points.shape[1] != 2 or len(points) < 3 or moving.shape != (len(points),)
            or margin.shape != (len(points),) or anchor.shape != (3,) or plane.shape != (3,)
            or uncertainty.shape != (len(points),) or np.any(uncertainty < 0.)
            or preferred.shape != (3,)
            or not all(np.all(np.isfinite(value)) for value in
                       (q, normal, points, margin, anchor, plane, preferred, uncertainty))
            or not math.isfinite(curvature) or abs(curvature) < .1
            or abs(np.linalg.norm(normal)-1.) > 1.e-6
            or abs(np.linalg.norm(plane)-1.) > 1.e-6
            or abs(normal[2]) > 1.e-6 or abs(plane[2]) > 1.e-6
            or np.any(margin < .03) or not .03 <= panel_margin < .3
            or not 0. < opening_speed <= .06 or not 1 <= depth <= 16
            or not 1 <= beam_width <= 32 or not 0. < interval <= 2.):
        raise ValueError('Invalid observed contact lookahead inputs')
    outline = kinematics.body_outline if hasattr(kinematics, 'body_outline') else None
    if outline is None:
        from .piper_actual_kinematics import PIPER_BODY_OUTLINE
        outline = PIPER_BODY_OUTLINE
    outline = np.asarray(outline, float)
    desired_heading=(contact_steady_heading(curvature,preferred[0])
                     if options.get('measured_heading_reference',False) else None)
    low, high = outline.min(axis=0), outline.max(axis=0)
    point, rotation = kinematics.forward(q)
    start_condition = kinematics.translation_condition_ratio(q)
    if start_condition < .20 or np.any(q < PIPER_JOINT_LIMITS[:, 0]+.20-1.e-6) or np.any(q > PIPER_JOINT_LIMITS[:, 1]-.20+1.e-6):
        raise ValueError('Lookahead requires current measured joint reserve')
    # A measured path curvature supplies this prediction centre. It is not a
    # simulator hinge location; prediction error is guarded during execution.
    hinge = point.copy()
    hinge[:2] += np.array([-normal[1], normal[0]])/curvature
    initial_gaps = np.linalg.norm(points-np.clip(points, low, high), axis=1)
    if min(initial_gaps) < .03:
        raise ValueError('Observed scene already violates hard chassis clearance')
    # Do not invent instant reserve when stopped inside a soft uncertainty
    # margin: permit only nondecreasing gap until that margin is recovered.
    required = np.minimum(margin, initial_gaps)
    initial_plane_gap = min((anchor[:2]-outline)@plane[:2])
    if initial_plane_gap < .03:
        raise ValueError('Observed panel already violates hard chassis clearance')
    required_plane = min(panel_margin, initial_plane_gap)
    actions = [(v, w) for v in (.06, .04, .02, 0., -.02)
               for w in (-.08, -.04, 0., .04, .08) if v or w]
    # State stores scene geometry in the predicted base frame. Each successor
    # is checked over the arc, not merely at its endpoint.
    initial = (q, point, rotation, normal, points, anchor, plane, hinge,
               0., float(min(initial_gaps)), start_condition, None, None,0.,())
    beam = [initial]
    evaluated = 0
    best = None
    for level in range(depth):
        successors = []
        for state in beam:
            # Predict commands after the SAME arm/scene/panel projection used
            # by execution. IK-only feasibility overestimates reachable motion.
            motion = np.column_stack((-(state[4]-state[7][:2])[:, 1],
                                      (state[4]-state[7][:2])[:, 0]))*curvature
            motion[~moving] = 0.
            scene_rows, scene_bounds, _ = scanned_body_constraints(
                state[4], outline, margin=margin, point_opening_motion=motion,
                motion_uncertainty=uncertainty)
            arm_rows, arm_bounds, _ = kinematics.contact_velocity_barriers(
                state[0], state[3], 0., contact_curvature=curvature)
            plane_anchor = state[1]+float((state[5]-state[1])@state[6])*state[6]
            projection = dict(margin=panel_margin, opening_speed=opening_speed,
                panel_anchor=plane_anchor, panel_normal=state[6],
                observed_curvature=curvature, arm_constraints=(arm_rows, arm_bounds),
                scene_constraints=(scene_rows, scene_bounds),
                limit_posture_drift=False, allow_posture_recovery=True,
                projection_cache={}, **options)
            projected_actions = {}
            for velocity in actions:
                evaluated += 1
                try:
                    v, w, _, _ = contact_posture_command(state[1], state[3],
                        preferred, outline, nominal_velocity=velocity,
                        prefer_nominal_recovery=velocity[0]<0., **projection)
                except ValueError:
                    continue
                cell = tuple(np.rint(np.array([v, w])/1.e-8).astype(int))
                distance = float(np.linalg.norm(np.array(velocity)-[v, w]))
                old = projected_actions.get(cell)
                if old is None or distance < old[0]:
                    projected_actions[cell] = (distance, velocity, (v, w))
            # Different requests often project onto the same constraint face.
            # Sweep that physical arc once, without weakening any guard.
            for _, velocity, (v, w) in projected_actions.values():
                value = _advance(kinematics, state[:13], (v, w), interval, opening_speed,
                    curvature, moving, required, outline, required_plane)
                if value is None:
                    continue
                if state[11] is None:
                    value = (*value[:11], velocity, (v, w))
                # Progress plus reserve selects a feasible detour even when
                # instantaneous distance to a future hand posture increases.
                hand_error = np.linalg.norm(value[1][:2]-preferred[:2])
                # Rank the whole predicted approach, not only a good final
                # posture that can perpetually be deferred by replanning.
                extension=max(0.,float(value[1][0]-preferred[0]))
                path_error=state[13]+hand_error+6.*extension
                arcs=(*state[14],dict(velocity=velocity,projected_velocity=(v,w),
                    tool_xyz=value[1].tolist(),normal_xyz=value[3].tolist()))
                value=(*value,path_error,arcs)
                relative_heading=(math.atan2(value[3][1],value[3][0])
                                  if desired_heading is not None else None)
                score = contact_prediction_score(value[8],hand_error,value[10],
                    relative_heading,desired_heading,path_error/(level+1))
                first = value[11]
                score -= .002*(first[0] < 0.)
                successors.append((score, value))
        if not successors:
            raise ValueError('No contact arc survives the full lookahead horizon')
        successors.sort(key=lambda item: item[0], reverse=True)
        # Quantized diversity prevents one arc family occupying every slot.
        beam = _diverse_beam(successors, beam_width)
        best = successors[0][1]
    if best is None:
        raise ValueError('No observed scene-clear contact lookahead arc')
    return ContactPlan(best[11], best[8], best[9], best[10], evaluated, best[12], interval,
                       best[14],0,curvature)


def _diverse_beam(successors, width):
    # Preserve clockwise, straight and counterclockwise first manoeuvres.
    # A necessary setup turn may temporarily increase soft hand-goal error.
    beam, cells, counts = [], set(), {}
    quota = max(1, width//3)
    for balanced in (True, False):
        for _, value in successors:
            group = int(np.sign(value[12][1]))
            cell = (group, *np.rint(np.r_[value[1][:2]/.025,
                math.atan2(value[3][1], value[3][0])/.10, value[8]/.05]).astype(int))
            if cell in cells or (balanced and counts.get(group, 0) >= quota):
                continue
            cells.add(cell)
            counts[group] = counts.get(group, 0)+1
            beam.append(value)
            if len(beam) >= width:
                return beam
    return beam


def _advance(arm, state, velocity, interval, request, curvature, moving,
             required, outline, required_plane):
    q, point, rotation, normal, points, anchor, plane, hinge, opened, gap, condition, first, projected = state
    v, w = velocity
    first = velocity if first is None else first
    projected = velocity if projected is None else projected
    # Substeps keep actual IK changes and the chassis sweep bounded.
    steps = max(4, int(math.ceil(interval/.2)))
    dt = interval/steps
    for _ in range(steps):
        supplied = max(0., float(np.array([v-w*point[1], w*point[0], 0.])@normal))
        u = min(request, supplied)
        turn, door_turn = w*dt, curvature*u*dt
        door_rotation = _rotate(door_turn)
        base_inverse = _rotate(-turn)
        shift = (np.array([v*dt, 0., 0.]) if abs(w) < 1.e-9 else
                 np.array([v/w*math.sin(turn), v/w*(1.-math.cos(turn)), 0.]))
        target = base_inverse@(hinge+door_rotation@(point-hinge)-shift)
        next_rotation = base_inverse@door_rotation@rotation
        solution = arm.solve(target, seed=q, target_rotation=next_rotation,
            position_tolerance_m=.001, orientation_tolerance_rad=.015,
            max_iterations=35, prefer_seed_solution=True, minimum_joint_margin_rad=.20,
            seed_only=True)
        if solution is None or np.max(abs(solution.positions-q)) > .14:
            return None
        q = solution.positions
        measured, measured_rotation = arm.forward(q)
        condition = min(condition, arm.translation_condition_ratio(q))
        if condition < .20:
            return None
        cloud = np.column_stack((points, np.zeros(len(points))))
        cloud[moving] = (cloud[moving]-hinge)@door_rotation.T+hinge
        points = ((cloud-shift)@base_inverse.T)[:, :2]
        gaps = np.linalg.norm(points-np.clip(points, outline.min(axis=0), outline.max(axis=0)), axis=1)
        if min(gaps) < .03 or np.any(gaps < required-1.e-5):
            return None
        anchor = base_inverse@(hinge+door_rotation@(anchor-hinge)-shift)
        plane = base_inverse@door_rotation@plane
        if min((anchor[:2]-outline)@plane[:2]) < required_plane-1.e-5:
            return None
        hinge = base_inverse@(hinge-shift)
        normal = base_inverse@door_rotation@normal
        point, rotation = measured, measured_rotation
        opened += abs(door_turn)
        gap = min(gap, float(min(gaps)))
    return (q, point, rotation, normal, points, anchor, plane, hinge,
            opened, gap, condition, first, projected)
