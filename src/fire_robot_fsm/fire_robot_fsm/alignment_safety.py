"""Check a short manual-approach segment in Nav2's inflated occupancy grid."""
import math


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
