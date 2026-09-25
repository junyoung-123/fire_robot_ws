"""Door rotation from measured depth planes; no simulator state or world poses."""
import math
import numpy as np


def rotation_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)/np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def fit_panel(points, anchor, rng, threshold=.008, min_points=100, prior=None):
    """Fit a vertical plane near an observed handle, rejecting occluding objects."""
    points = np.asarray(points, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[np.linalg.norm(points-anchor, axis=1) < .95]
    if len(points) < min_points:
        raise ValueError('too_few_depth_points_near_handle')
    best = None
    # A prior is only a RANSAC hypothesis: current depth must support it.
    if prior is not None:
        normal, center = prior
        for _ in range(3):
            mask = np.abs((points-center)@normal) < threshold
            if int(mask.sum()) < min_points:
                break
            inliers = points[mask]
            center = inliers.mean(axis=0)
            _, _, vh = np.linalg.svd(inliers-center, full_matrices=False)
            normal = vh[-1]
            if abs(normal[2]) <= .15 and abs(np.dot(anchor-center, normal)) <= .13:
                best = int(mask.sum()), mask
    for _ in range(64):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1.e-6:
            continue
        normal /= norm
        if abs(normal[2]) > .15:
            continue
        if abs(np.dot(anchor-sample[0], normal)) > .13:
            continue
        mask = np.abs((points-sample[0])@normal) < threshold
        count = int(mask.sum())
        if best is None or count > best[0]:
            best = count, mask
    if best is None or best[0] < min_points or best[0]/len(points) < .35:
        raise ValueError('no_supported_vertical_panel')
    inliers = points[best[1]]
    center = inliers.mean(axis=0)
    _, singular, vh = np.linalg.svd(inliers-center, full_matrices=False)
    normal = vh[-1]
    residual = float(np.sqrt(np.mean(((inliers-center)@normal)**2)))
    if residual > threshold or abs(normal[2]) > .15:
        raise ValueError('invalid_plane_residual_or_tilt')
    support=float(singular[1]/math.sqrt(len(inliers)))
    # A narrow but well sampled patch can constrain orientation. Bound the
    # noise-to-baseline ratio, cap effective samples, and cross-check spatial
    # subsets instead of accepting/rejecting solely by physical patch width.
    uncertainty=3*max(residual,.001)/(math.sqrt(min(len(inliers),64))*max(support,1.e-9))
    coords=(inliers-center)@vh[:2].T
    halves=coords>np.median(coords,axis=0)
    for major in (False,True):
        for minor in (False,True):
            subset=inliers[~((halves[:,0]==major)&(halves[:,1]==minor))]
            if len(subset)<75:
                raise ValueError('too_few_points_for_spatial_cross_check')
            _,_,axes=np.linalg.svd(subset-subset.mean(axis=0),full_matrices=False)
            disagreement=math.acos(float(np.clip(abs(axes[-1]@normal),0.,1.)))
            uncertainty=max(uncertainty,disagreement)
    if uncertainty>math.radians(2.):
        raise ValueError('panel_orientation_uncertain')
    return dict(normal=normal, center=center, residual_m=residual,
                inliers=len(inliers), inlier_ratio=len(inliers)/len(points),
                support_std_m=support, orientation_uncertainty_rad=uncertainty)


class PlaneRotation:
    """Track the unoriented plane continuously beyond 90 degrees from baseline."""
    def __init__(self, baseline_samples=5):
        self.required = baseline_samples
        self.baseline = []
        self.reference = None
        self.previous = None
        self.stamp = None

    def update(self, normal, stamp):
        angle = math.atan2(float(normal[1]), float(normal[0]))
        reference = self.previous if self.previous is not None else angle
        angle += round((reference-angle)/math.pi)*math.pi
        if self.stamp is not None:
            dt = stamp-self.stamp
            if dt <= 0:
                raise ValueError('non_increasing_depth_stamp')
            if dt > .75 and abs(angle-reference) > .12:
                raise ValueError('unobservable_rotation_during_gap')
            if abs(angle-reference) > .035+.45*dt:
                raise ValueError('plane_orientation_jump')
        if self.reference is None:
            self.baseline.append(angle)
            self.baseline = self.baseline[-self.required:]
            if len(self.baseline) == self.required:
                if np.ptp(self.baseline) > math.radians(2):
                    raise ValueError('baseline_not_stable')
                self.reference = float(np.median(self.baseline))
        self.previous, self.stamp = angle, stamp
        return None if self.reference is None else angle-self.reference
