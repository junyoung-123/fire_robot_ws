#!/usr/bin/env python3
"""An LK seed from one camera must never advance on another camera's frame."""
from types import SimpleNamespace
import numpy as np
import fire_robot_perception.door_detection_node as module

class Tracker:
    def __init__(self):
        self.seeds = []
        self.advances = 0
    def seed(self, image, *args):
        self.seeds.append(int(image[0,0,0]))
    def advance(self, *args):
        self.advances += 1
        return None

def main():
    module.HandleImageTracker = Tracker
    subject = SimpleNamespace(_detect_handle_enabled=True,
        _handle_registered_depth=True, _feedback_handle_trackers={},
        _handle_yolo_fallback_hsv=False)
    image = np.ones((240,320,3), dtype=np.uint8)
    detected = SimpleNamespace(method='yolo:primary:item', confidence=.8, bbox=(10,10,30,20))
    subject._detect_yolo_handle_observation = lambda *args: detected
    module.DoorDetectionNode._detect_handle_observation(subject, image, image, 0,0,320,240,'blue',image.shape,'front')
    subject._detect_yolo_handle_observation = lambda *args: None
    module.DoorDetectionNode._detect_handle_observation(subject, image*2, image, 0,0,320,240,'blue',image.shape,'left')
    module.DoorDetectionNode._detect_handle_observation(subject, image, image, 0,0,320,240,'blue',image.shape,'front')
    front, left = (subject._feedback_handle_trackers[n] for n in ('front','left'))
    assert front is not left
    assert front.seeds == [1] and left.seeds == []
    assert front.advances == left.advances == 1
    print('PASS: camera-specific primary YOLO tracking histories')

if __name__ == '__main__':
    main()
