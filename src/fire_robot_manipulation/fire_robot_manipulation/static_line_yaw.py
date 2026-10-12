"""Yaw-only fallback from long, measured static lines; no environment priors."""
from dataclasses import dataclass
import math
import numpy as np


def axis_difference(angle):
    return .5*math.atan2(math.sin(2.*angle), math.cos(2.*angle))


@dataclass
class Line:
    angle: float
    uncertainty: float
    residual: float
    count: int
    span: float


def measured_lines(points):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.all(np.isfinite(points)):
        raise ValueError('Invalid measured line points')
    remaining=points.copy()
    rng=np.random.default_rng(8431)
    lines=[]
    for _ in range(6):
        if len(remaining)<35:
            break
        pairs=rng.integers(0,len(remaining),(200,2))
        d=remaining[pairs[:,1]]-remaining[pairs[:,0]]
        length=np.linalg.norm(d,axis=1)
        pairs,d=pairs[length>.8],d[length>.8]
        if not len(pairs):
            break
        normals=np.column_stack((-d[:,1],d[:,0]))/np.linalg.norm(d,axis=1)[:,None]
        distance=abs(remaining@normals.T-np.einsum('ij,ij->i',remaining[pairs[:,0]],normals))
        masks=distance<.10
        mask=masks[:,np.argmax(masks.sum(axis=0))]
        if mask.sum()<35:
            break
        for _ in range(4):
            selected=remaining[mask]
            center=selected.mean(axis=0)
            values,vectors=np.linalg.eigh(np.cov((selected-center).T,bias=True))
            normal=vectors[:,0]
            updated=abs((remaining-center)@normal)<.10
            if updated.sum()<35:
                break
            if np.array_equal(updated,mask):
                break
            mask=updated
        selected=remaining[mask]
        centered=selected-selected.mean(axis=0)
        values,vectors=np.linalg.eigh(centered.T@centered/len(selected))
        axis=vectors[:,1]
        projection=centered@axis
        span=float(np.quantile(projection,.95)-np.quantile(projection,.05))
        residual=math.sqrt(max(0.,float(values[0])))
        uncertainty=3.*max(.005,residual)/math.sqrt(max(1.e-12,(len(selected)-2)*values[1]))
        if span>=1.5 and residual<=.055 and uncertainty<=math.radians(1.):
            lines.append(Line(math.atan2(axis[1],axis[0]),uncertainty,residual,len(selected),span))
        remaining=remaining[~mask]
    return lines


class StaticLineYaw:
    """Recover heading only, never translation, from measured long-wall axes.

    An isolated short panel/box cannot supply the required 1.5m line span.
    The wheel-seeded 2-degree continuity gate is unchanged. Contradictory line
    directions invalidate the fallback rather than selecting a convenient one.
    """
    def __init__(self,reference):
        self.lines=measured_lines(reference)

    def estimate(self,cloud,seed_yaw):
        if not math.isfinite(seed_yaw) or not self.lines:
            raise ValueError('No measured long static reference line')
        candidates=[]
        for current in measured_lines(cloud):
            matches=[]
            for reference in self.lines:
                delta=axis_difference(reference.angle-current.angle-seed_yaw)
                confidence=reference.uncertainty+current.uncertainty
                if abs(delta)<=math.radians(2.) and confidence<=math.radians(1.5):
                    matches.append((confidence,delta,reference))
            if matches:
                confidence,delta,reference=min(matches,key=lambda r:r[0])
                candidates.append((delta,confidence,current))
        if not candidates:
            raise ValueError('No continuous measured static yaw support')
        deltas=np.array([r[0] for r in candidates])
        weights=np.array([1./max(r[1],math.radians(.1))**2 for r in candidates])
        mean=float(np.average(deltas,weights=weights))
        bound=max(r[1]+abs(r[0]-mean) for r in candidates)
        if bound>math.radians(1.5):
            raise ValueError('Independent measured static line headings disagree')
        return dict(yaw=math.atan2(math.sin(seed_yaw+mean),math.cos(seed_yaw+mean)),
            yaw_uncertainty_rad=bound,residual_rms_m=max(r[2].residual for r in candidates),
            inliers=sum(r[2].count for r in candidates),line_count=len(candidates),
            measured_spans_m=[r[2].span for r in candidates],translation_observed=False)
