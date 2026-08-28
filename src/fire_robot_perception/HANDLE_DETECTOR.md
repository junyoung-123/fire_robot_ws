# Lever-handle detector

This package adds a dedicated one-class YOLO detector for door lever handles.
The door detector still finds and classifies the door; the handle detector
selects the highest-confidence `lever_handle` whose center lies inside that
door. Its horizontal pixel direction is used for `DoorInfo.handle_position`.
If the model or a matching detection is unavailable, the previous door-center
fallback remains active.

## Training result (2026-08-28)

- Source: Open Images V7 `Door handle` bounding boxes (`/m/03c7gz`)
- Split: 400 train images, 100 independent test-split images used as validation
- Labels: 657 boxes, one class named `lever_handle`
- Model: YOLOv8s, 640 px, batch 8, 80 epochs
- Precision: 0.6351
- Recall: 0.3914
- mAP50: 0.3599
- mAP50-95: 0.2388
- Installed weight: `models/handle_best.pt`

The initial public-data model is suitable for integration validation. Recall is
the main limitation; collect close-range images from the robot camera and
fine-tune before relying on it for final field operation.

## Reproduce

Open Images metadata must exist under `~/fiftyone/open-images-v7` for the
`train` and `test` splits. The preparation script downloads canonical
images from the official public bucket and writes YOLO labels.

```bash
python3 scripts/prepare_handle_dataset.py \
  --train-count 400 --val-count 100 --workers 16

python3 scripts/train_handle_detector.py \
  --dataset datasets/handle_detection/dataset.yaml \
  --model yolov8s.pt --epochs 80 --batch 8 --device 0 --install
```

Launch arguments `handle_model_path`, `handle_confidence_threshold`, and
`handle_height_m` configure runtime inference. Simulation and real-robot
bringup default to the installed `models/handle_best.pt`.
