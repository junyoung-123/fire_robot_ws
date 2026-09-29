"""Associate a green observation with a LiDAR-observed, bounded open passage."""
import math
import statistics


def find_aperture(ranges, angle_min, increment, range_min, range_max, anchor,
                  min_width=.75, max_width=2.5, association_radius=2.0):
    # Positive infinity is the configured LiDAR's no-return measurement.
    # NaN, negative infinity and missing rays never establish free space.
    no_return = [math.isinf(r) and r > 0 for r in ranges]
    candidates = []
    start = None
    for i, clear in enumerate(no_return + [False]):
        if clear and start is None:
            start = i
        if clear or start is None:
            continue
        end, begin = i, start
        start = None
        if begin < 3 or end + 2 >= len(ranges) or end - begin < 3:
            continue
        sides = [ranges[begin-3:begin], ranges[end:end+3]]
        if any(not all(math.isfinite(r) and range_min < r < range_max for r in side)
               for side in sides):
            continue
        points = []
        for index, side in zip((begin-1, end), sides):
            r = statistics.median(side)
            a = angle_min + index*increment
            points.append((r*math.cos(a), r*math.sin(a)))
        (ax, ay), (bx, by) = points
        width = math.hypot(bx-ax, by-ay)
        if not min_width <= width <= max_width:
            continue
        x, y = (ax+bx)/2, (ay+by)/2
        distance = math.hypot(x-anchor[0], y-anchor[1])
        if distance > association_radius:
            continue
        nx, ny = (by-ay)/width, -(bx-ax)/width
        if nx*x + ny*y < 0:
            nx, ny = -nx, -ny
        # Reject grazing views: the two sides do not yet constrain a crossing.
        if (nx*x+ny*y) / max(math.hypot(x, y), .01) < .6:
            continue
        candidates.append((distance, (x, y, math.atan2(ny, nx), width)))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]
