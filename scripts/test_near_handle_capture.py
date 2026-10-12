"""Regression on a real recorded near-field frame; no synthetic handle box."""
from threading import Lock
from types import SimpleNamespace
import cv2
from ultralytics import YOLO
from fire_robot_perception.door_detection_node import DoorDetectionNode

image=cv2.imread('/mnt/c/Users/황준영/Documents/졸업작품/tmp/world1_full_20261003/r32_alignment.jpg')
camera=cv2.resize(image[45:455,987:1533],(320,240))
model=YOLO('src/fire_robot_perception/models/handle_best_v3_CANDIDATE.pt')
node=SimpleNamespace(
    _handle_model=model,_handle_yolo_class_ids=(),_handle_fallback_model=None,
    _handle_yolo_roi_expand_x_px=0,_handle_yolo_roi_expand_y_px=0,
    _handle_yolo_full_frame=True,_handle_registered_depth=True,
    _handle_confidence_threshold=.25,_handle_yolo_imgsz=320,
    _handle_yolo_min_area_px=12,_handle_yolo_max_area_ratio=.25,
    _torch_num_threads=2,_yolo_lock=Lock(),_log_handle_detections=False,
    _model_class_name=lambda model,index: model.names[index])
result=DoorDetectionNode._detect_yolo_handle_observation(
    node,camera,0,0,320,240,camera.shape)
assert result is not None and result.method.startswith('yolo:primary')
assert result.confidence >= .45
assert 0 <= result.x < 320 and 0 <= result.y < 240
print(result)
