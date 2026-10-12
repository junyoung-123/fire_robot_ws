#!/usr/bin/env python3
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch
from fire_robot_perception.door_detection_node import DoorDetectionNode

processed = []
subject = SimpleNamespace(_pending_images={}, _pending_image_lock=Lock(),
    _image_processing_lock=Lock(), _last_image_process_time={},
    _image_interval_for_source=lambda name: .25 if name=='front' else 1.2,
    bridge=SimpleNamespace(imgmsg_to_cv2=lambda msg, encoding: msg.value),
    _process=lambda image, header, source: processed.append((source.name, image)))
front = SimpleNamespace(name='front')
left = SimpleNamespace(name='front_left')
right = SimpleNamespace(name='front_right')
for source, value in ((left, 1), (right, 1), (front, 1), (front, 2)):
    DoorDetectionNode.image_callback(subject, SimpleNamespace(value=value, header=None), source)
with patch('fire_robot_perception.door_detection_node.time.monotonic', return_value=100.):
    DoorDetectionNode.process_pending_image(subject)
    DoorDetectionNode.process_pending_image(subject)
    DoorDetectionNode.process_pending_image(subject)
assert processed == [('front', 2), ('front_left', 1), ('front_right', 1)]
assert not subject._image_processing_lock.locked()
assert not subject._pending_images
print('PASS: newest-only camera frames; frontal priority; side cameras serviced; lock released')
