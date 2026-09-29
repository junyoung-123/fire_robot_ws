"""Local wall normal from finite LiDAR returns near a visual door observation."""
import math
import numpy as np


def fit_wall_normal(ranges, angle_min, increment, range_min, range_max, anchor):
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + np.arange(len(ranges))*increment
    valid = np.isfinite(ranges) & (ranges > range_min) & (ranges < range_max)
    points = np.column_stack((ranges[valid]*np.cos(angles[valid]),
                              ranges[valid]*np.sin(angles[valid])))
    anchor = np.asarray(anchor, dtype=float)
    points = points[np.linalg.norm(points-anchor, axis=1) < .9]
    if len(points) < 12:
        return None
    # Deterministic RANSAC rejects an isolated foreground obstacle or jamb.
    samples = points[np.linspace(0, len(points)-1, min(16, len(points))).astype(int)]
    best = None
    for i, a in enumerate(samples):
        for b in samples[i+1:]:
            vector = b-a
            length = np.linalg.norm(vector)
            if length < .45:
                continue
            n = np.array([-vector[1], vector[0]])/length
            mask = np.abs((points-a) @ n) < .07
            if mask.sum() < 12:
                continue
            inliers = points[mask]
            center = inliers.mean(axis=0)
            _, _, axes = np.linalg.svd(inliers-center, full_matrices=False)
            n = axes[-1]
            if n @ center < 0:
                n = -n
            span = np.ptp((inliers-center) @ axes[0])
            rmse = np.sqrt(np.mean(((inliers-center) @ n)**2))
            offset = abs((anchor-center) @ n)
            if span < .55 or rmse > .055 or offset > .35:
                continue
            if n @ anchor / max(np.linalg.norm(anchor), .01) < .65:
                continue
            # Nearer returns can occlude the wall. Compare support only with
            # points at/behind its depth, not the denser foreground surface.
            residuals = (points-center) @ n
            visible = residuals >= -.10
            count = int((np.abs(residuals) < .07).sum())
            if count < max(12, visible.sum()*.65):
                continue
            score = count / (1.0 + 10.0*offset)
            if best is None or score > best[0]:
                best = score, n
    if best is None:
        return None
    return math.atan2(best[1][1], best[1][0])
