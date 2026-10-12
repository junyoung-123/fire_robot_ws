"""Expected static chassis echoes from the supplied robot CAD, not the world."""
import math
import struct
import numpy as np


def horizontal_sections(path, z):
    content = path.read_bytes()
    count = struct.unpack_from('<I', content, 80)[0]
    segments = []
    for index in range(count):
        values = struct.unpack_from('<12fH', content, 84+50*index)
        triangle = np.asarray(values[3:12]).reshape(3, 3)
        points = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            if (a[2] <= z < b[2]) or (b[2] <= z < a[2]):
                points.append((a+(b-a)*(z-a[2])/(b[2]-a[2]))[:2])
        if len(points) == 2:
            segments.append(points)
    return np.asarray(segments, dtype=float)


def expected_echo(segments, angles, sensor_xy, minimum_distance=.01):
    result = np.full(len(angles), np.inf)
    if not len(segments):
        return result
    start = segments[:, 0]-np.asarray(sensor_xy)
    edge = segments[:, 1]-segments[:, 0]
    for index, angle in enumerate(angles):
        ray = np.array([math.cos(angle), math.sin(angle)])
        denominator = ray[0]*edge[:, 1]-ray[1]*edge[:, 0]
        valid = np.abs(denominator) > 1e-10
        distance = np.full(len(edge), np.inf)
        fraction = np.full(len(edge), np.inf)
        distance[valid] = (start[valid, 0]*edge[valid, 1]-start[valid, 1]*edge[valid, 0])/denominator[valid]
        fraction[valid] = (start[valid, 0]*ray[1]-start[valid, 1]*ray[0])/denominator[valid]
        hits = valid & (distance >= minimum_distance) & (fraction >= 0) & (fraction <= 1)
        if hits.any():
            result[index] = distance[hits].min()
    return result


def mask_echoes(ranges, expected, tolerance=.17):
    ranges = np.asarray(ranges, dtype=float).copy()
    expected=np.asarray(expected,dtype=float)
    if ranges.shape!=expected.shape or not math.isfinite(tolerance) or tolerance<0.:
        raise ValueError('Invalid robot CAD ray mask')
    valid = np.isfinite(expected)
    self_hit = np.zeros(len(ranges), dtype=bool)
    # An opaque CAD surface also occludes longer/no-return rays. They cannot
    # measure a world obstacle behind the robot, nor clear free space there.
    # A closer obstacle outside the self-echo tolerance remains observable.
    self_hit[valid] = (~np.isfinite(ranges[valid])
                      | (ranges[valid]>=expected[valid]-tolerance))
    # NaN is unobserved, not a free-space ray through an occluding robot part.
    ranges[self_hit] = np.nan
    return ranges, int(self_hit.sum())


def mask_confirmed_self_noise(cleaned, expected, history, tolerance=.17):
    """Keep a CAD-confirmed occluding ray unobserved through isolated noise.

    Large innovations or three repeated nearer echoes remain obstacles. This
    never marks free space behind the robot's opaque fixed-link geometry.
    """
    result=np.asarray(cleaned,float).copy()
    values=np.asarray(history,float)
    expected=np.asarray(expected,float)
    if values.ndim!=2 or values.shape[1]!=len(result) or expected.shape!=result.shape:
        raise ValueError('Invalid self-echo history')
    if len(values)<7:
        return result
    selected=np.isfinite(expected)&np.isfinite(result)&np.isfinite(values).all(axis=0)
    if not selected.any():
        return result
    samples=values[:,selected]
    median=np.median(samples,axis=0)
    sigma=np.maximum(.005,1.4826*np.median(abs(samples-median),axis=0))
    threshold=expected[selected]-tolerance
    confirmed=(median>=threshold)&(samples[-1]>=median-4.*sigma-.005)
    repeated=(samples[-3:]<threshold).all(axis=0)
    indexes=np.flatnonzero(selected)[confirmed&~repeated]
    result[indexes]=np.nan
    return result
