#!/usr/bin/env python3
"""Train and optionally install the one-class lever-handle YOLO model."""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', default='yolov8s.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='door_handle_detector')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--install-path',
                        default='src/fire_robot_perception/models/handle_best.pt')
    return parser.parse_args()


def main():
    from ultralytics import YOLO

    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    model = YOLO(args.model)
    model.train(
        data=str(dataset), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers,
        device=args.device or None, project=args.project, name=args.name,
        patience=args.patience, hsv_h=0.010, hsv_s=0.35, hsv_v=0.35,
        degrees=4.0, translate=0.08, scale=0.35, fliplr=0.5,
        mosaic=0.6, mixup=0.05, save=True, plots=True, val=True)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / 'weights' / 'best.pt'
    metrics = model.val(data=str(dataset), split='val')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'Best model: {best.resolve()}')
    if args.install:
        target = Path(args.install_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        print(f'Installed: {target}')


if __name__ == '__main__':
    main()
