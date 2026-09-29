"""Dominant unoriented wall direction from occupied cells, without world geometry."""
import math

import cv2
import numpy as np


def estimate_wall_axis(occupied, resolution):
    image = np.asarray(occupied, dtype=np.uint8) * 255
    lines = cv2.HoughLinesP(
        image, 1, math.pi / 180, threshold=max(12, int(0.75 / resolution)),
        minLineLength=max(8, int(1.2 / resolution)),
        maxLineGap=max(1, int(0.20 / resolution)))
    if lines is None:
        return None
    segments = lines[:, 0, :].astype(float)
    dx, dy = segments[:, 2]-segments[:, 0], segments[:, 3]-segments[:, 1]
    angles = np.arctan2(dy, dx)
    lengths = np.hypot(dx, dy)
    delta = angles[:, None] - angles[None, :]
    distances = np.abs(np.arctan2(np.sin(2*delta), np.cos(2*delta))) / 2
    support = distances <= math.radians(6)
    cluster = support[np.argmax(support @ lengths)]
    return 0.5 * math.atan2(
        float(np.sum(lengths[cluster] * np.sin(2*angles[cluster]))),
        float(np.sum(lengths[cluster] * np.cos(2*angles[cluster]))))
