"""Short-lived optical-flow tracking seeded by a confident YOLO handle.

This is an image measurement, not a new YOLO detection or a saved 3D point.
Forward/backward consistency and affine inliers reject lost/occluded tracks.
"""
import cv2
import numpy as np


class HandleImageTracker:
    def __init__(self):
        self.gray = None
        self.points = None
        self.box = None
        self.seed_time = None
        self.confidence = 0.0

    def seed(self, image, box, confidence, now):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        x1, y1, x2, y2 = map(int, box)
        mask[max(0, y1):y2, max(0, x1):x2] = 255
        points = cv2.goodFeaturesToTrack(gray, 40, .01, 3, mask=mask)
        if points is None or len(points) < 4:
            return False
        self.gray, self.points = gray, points
        self.box = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        self.seed_time, self.confidence = now, confidence
        return True

    def advance(self, image, now):
        if self.gray is None or not 0 <= now - self.seed_time <= 15.0:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.gray.shape:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, winSize=(21, 21), maxLevel=3)
        if nxt is None:
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, nxt, None, winSize=(21, 21), maxLevel=3)
        if back is None:
            return None
        h,w=gray.shape
        valid = ((status.ravel() != 0) & (back_status.ravel() != 0)
                 & (np.linalg.norm(back - self.points, axis=2).ravel() < 1.0)
                 & (nxt[:,0,0]>=0.) & (nxt[:,0,0]<w)
                 & (nxt[:,0,1]>=0.) & (nxt[:,0,1]<h))
        if np.count_nonzero(valid) < 4 or np.mean(valid) < .7:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            self.points[valid], nxt[valid], method=cv2.RANSAC,
            ransacReprojThreshold=1.5)
        if matrix is None or inliers is None or np.mean(inliers) < .75:
            return None
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not .8 <= scale <= 1.25:
            return None
        box = cv2.transform(self.box[None], matrix)[0]
        centre=box.mean(axis=0)
        if (not np.all(np.isfinite(box)) or np.any(centre < 2)
                or centre[0]>=w-2 or centre[1]>=h-2):
            return None
        self.gray, self.points, self.box = gray, nxt[valid], box
        confidence = min(self.confidence, float(np.mean(valid)), float(np.mean(inliers)))
        return tuple(np.round(box).astype(int).ravel()), confidence

    def confirm_partial_primary(self,box,confidence,now):
        """Refresh object identity without moving its tracked grasp centre."""
        if (self.box is None or self.gray is None or self.seed_time is None
                or not 0.<=now-self.seed_time<=15. or not .45<=confidence<=1.):
            return False
        proposed=np.asarray(box,float).reshape(2,2)
        h,w=self.gray.shape
        visible=np.clip(self.box,[0.,0.],[float(w),float(h)])
        size=proposed[1]-proposed[0]
        intersection=np.maximum(0.,np.minimum(visible[1],proposed[1])
                                -np.maximum(visible[0],proposed[0]))
        if (not np.all(np.isfinite(proposed)) or np.any(size<=0.)
                or np.prod(intersection)<.7*np.prod(size)):
            return False
        self.seed_time=now
        self.confidence=min(self.confidence,confidence)
        return True
