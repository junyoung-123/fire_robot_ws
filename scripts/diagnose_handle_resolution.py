"""Compare inference settings on recorded, unmodified camera images."""
import json
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
model = YOLO(ROOT / 'src/fire_robot_perception/models/handle_best_v3_CANDIDATE.pt')
rows = []
for run in ('observed_angle_r19', 'observed_angle_r20'):
    source = next((ROOT / 'artifacts/validation' / run).glob('physical_contact*'))
    for size in (320, 416, 512, 640, 800, 960):
        result = model(str(source / 'after_front_raw.png'), imgsz=size,
                       conf=.20, verbose=False)[0]
        rows.append(dict(run=run, size=size, detections=[
            dict(confidence=float(b.conf[0]), xyxy=b.xyxy[0].tolist())
            for b in result.boxes]))
print(json.dumps(rows, indent=2))
