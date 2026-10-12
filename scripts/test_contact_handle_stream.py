#!/usr/bin/env python3
"""The close-contact stream cannot turn HSV or missing depth into YOLO proof."""
from types import SimpleNamespace
import numpy as np
from std_msgs.msg import Header
from fire_robot_perception.door_detection_node import DoorDetectionNode


def main():
    emitted = []
    header = Header(frame_id='camera_color_optical_frame')
    header.stamp.sec = 12
    subject = SimpleNamespace(_frame='base_link',
        contact_handle_pub=SimpleNamespace(publish=emitted.append),
        get_logger=lambda: SimpleNamespace(info=lambda *args, **kwargs: None))
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for method, xyz, expected in (
            ('hsv', (.6, .1, .83), 0),
            ('yolo:fallback:item', (.6, .1, .83), 0),
            ('yolo:primary:item', None, 0),
            ('yolo:primary:item', (.6, .1, .83), 1),
            ('track:primary_yolo', (.6, .1, .83), 1)):
        emitted.clear()
        subject._detect_handle_observation = lambda *args, **kwargs: SimpleNamespace(
            method=method, confidence=.8, x=160, y=120, bbox=(140, 110, 180, 130))
        subject._registered_handle_point = lambda *args: xyz
        DoorDetectionNode._publish_contact_handle(subject, image, image, header,
                                                  SimpleNamespace(name='front'), None)
        assert len(emitted) == expected
        assert header.frame_id == 'camera_color_optical_frame'
        if emitted:
            msg = emitted[0]
            assert msg.handle_detected
            assert msg.handle_position.header.frame_id == 'base_link'
            assert msg.handle_position.header.stamp.sec == 12
            assert msg.handle_detection_method.endswith(':registered_depth')
    print('PASS: primary YOLO/track only, measured depth, unchanged camera header')


if __name__ == '__main__':
    main()
