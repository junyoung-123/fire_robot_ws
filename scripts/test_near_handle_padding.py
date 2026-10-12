#!/usr/bin/env python3
"""A padded inference must still return original camera pixels and depth inputs."""
from types import SimpleNamespace
from threading import Lock
import numpy as np
import fire_robot_perception.door_detection_node as module


def main():
    calls = []
    class Box:
        cls = 0
        conf = [.8]
        xyxy = [[150, 170, 330, 250]]
    def inference(model, sample, threads, **kwargs):
        calls.append(sample.shape)
        return [SimpleNamespace(boxes=[] if len(calls) == 1 else [Box()])]
    module.bounded_yolo_inference = inference
    subject = SimpleNamespace(
        _handle_model=object(), _handle_fallback_model=None,
        _handle_yolo_class_ids={0}, _handle_yolo_roi_expand_x_px=0,
        _handle_yolo_roi_expand_y_px=0, _handle_yolo_full_frame=True,
        _handle_confidence_threshold=.3, _handle_yolo_imgsz=320,
        _handle_yolo_min_area_px=3, _handle_yolo_max_area_ratio=.25,
        _yolo_lock=Lock(), _torch_num_threads=1, _handle_registered_depth=True,
        _log_handle_detections=False, _model_class_name=lambda *args: 'item')
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    observed = module.DoorDetectionNode._detect_yolo_handle_observation(
        subject, image, 0, 0, 320, 240, image.shape)
    assert calls == [(240, 320, 3), (400, 480, 3)], calls
    assert observed.bbox == (70, 90, 250, 170), observed
    assert (observed.x, observed.y) == (160, 130)
    assert observed.method == 'yolo:primary:item'
    assert not image.any()
    print('PASS: near-field primary inference restores original RGB coordinates')


if __name__ == '__main__':
    main()
