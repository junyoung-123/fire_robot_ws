#!/usr/bin/env python3
"""
Train a YOLOv8 door-handle detector.

Expected dataset:
  - YOLO format dataset.yaml
  - one class named lever_handle, handle, door_handle, knob, or lever

Example:
  python3 train_handle_detector.py \
    --dataset ~/datasets/door_handle_detection/dataset.yaml \
    --model yolov8s.pt \
    --epochs 80 \
    --imgsz 640 \
    --batch 16 \
    --install

After installation the runtime default path is:
  src/fire_robot_perception/models/handle_best_v2.pt
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='Path to YOLO dataset.yaml')
    parser.add_argument('--model', default='yolov8s.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
                        help='Base YOLO model')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='',
                        help='Training device: empty=auto, "0"=GPU0, "cpu"=CPU')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='door_handle_detector')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--install', action='store_true',
                        help='Copy best.pt into fire_robot_perception/models')
    parser.add_argument('--install_path', default='',
                        help='Override install destination for handle_best_v2.pt')
    return parser.parse_args()


def default_install_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'models' / 'handle_best_v2.pt'


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print('ERROR: ultralytics is not installed.')
        print('Install: pip install ultralytics')
        return 1

    args = parse_args()
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f'ERROR: dataset.yaml not found: {dataset_path}')
        return 1

    if args.resume:
        last_ckpt = Path(args.project) / args.name / 'weights' / 'last.pt'
        model_path = str(last_ckpt if last_ckpt.exists() else args.model)
    else:
        model_path = args.model

    print(f'Loading model: {model_path}')
    model = YOLO(model_path)

    print('Training door-handle detector:')
    print(f'  dataset: {dataset_path}')
    print(f'  epochs: {args.epochs}, batch: {args.batch}, imgsz: {args.imgsz}')
    print(f'  output: {args.project}/{args.name}')

    model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        hsv_h=0.010,
        hsv_s=0.35,
        hsv_v=0.35,
        degrees=4.0,
        translate=0.08,
        scale=0.35,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.05,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    save_dir = Path(getattr(model.trainer, 'save_dir',
                            Path(args.project) / args.name))
    best_model_path = save_dir / 'weights' / 'best.pt'
    if not best_model_path.exists():
        best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'

    print('\nValidation:')
    val_results = model.val(data=str(dataset_path), split='val')
    print(f'  mAP50:    {val_results.box.map50:.4f}')
    print(f'  mAP50-95: {val_results.box.map:.4f}')
    print(f'  Precision:{val_results.box.mp:.4f}')
    print(f'  Recall:   {val_results.box.mr:.4f}')

    print(f'\nBest model: {best_model_path.resolve()}')
    if args.install:
        install_path = Path(args.install_path).expanduser() if args.install_path else default_install_path()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_path, install_path)
        print(f'Installed: {install_path.resolve()}')

    print('\nROS2 usage:')
    print('  ros2 launch fire_robot_bringup simulation.launch.py \\')
    print(f'    handle_model_path:={best_model_path.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
