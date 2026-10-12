#!/usr/bin/env python3
"""Offline diagnostic on the captured camera view, not a control input."""
import json
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

ws = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('frame', nargs='?', default='artifacts/full_world1_r6/evidence/frames/000135.jpg')
args = parser.parse_args()
frame = cv2.imread(str(ws/args.frame))
image = cv2.resize(frame[45:455, 987:1533], (320, 240))
model = YOLO(str(ws/'src/fire_robot_perception/models/handle_best_v3_CANDIDATE.pt'))
for padding in (0, 80, 160):
    sample = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(114,114,114))
    for size in (320, 640):
        prediction = model(sample, conf=.05, imgsz=size, verbose=False)[0]
        print(json.dumps({'padding': padding, 'imgsz': size, 'boxes': prediction.boxes.xyxy.tolist(),
                          'confidence': prediction.boxes.conf.tolist()}))
        cv2.imwrite(str(ws/'artifacts'/f'offline_close_handle_{padding}_{size}.jpg'), prediction.plot())
