"""Sensor-independent guards for bounded, feedback-driven contact motion."""

import math
import numpy as np


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
