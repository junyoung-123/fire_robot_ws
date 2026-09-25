"""Confidence hysteresis only for an already acquired, measured 3D handle."""
from collections import deque
import math
import numpy as np


class ConfirmedHandleTrack:
    def __init__(self, acquire=.45, track=.20, max_shift=.10, spread_limit=.03):
        self.acquire,self.minimum=acquire,track
        self.max_shift,self.spread_limit=max_shift,spread_limit
        self.anchor=None
        self.last_stamp=None
        self.samples=deque(maxlen=5)

    def update(self, xyz, confidence, stamp):
        point=np.asarray(xyz,float)
        if (point.shape!=(3,) or not np.all(np.isfinite(point))
                or not math.isfinite(confidence) or not math.isfinite(stamp)
                or confidence<self.minimum
                or (self.last_stamp is not None and stamp<=self.last_stamp)):
            return False
        if self.anchor is None:
            if confidence<self.acquire:
                return False
        elif np.linalg.norm(point-self.anchor)>self.max_shift:
            self.samples.clear()
            return False
        self.last_stamp=stamp
        self.samples.append((stamp,point))
        recent=[p for t,p in self.samples if stamp-t<=3.]
        if confidence>=self.acquire:
            self.anchor=point.copy()
            return True
        if len(recent)<3:
            return False
        center=np.median(recent,axis=0)
        if max(np.linalg.norm(p-center) for p in recent)>self.spread_limit:
            return False
        self.anchor=center
        return True
