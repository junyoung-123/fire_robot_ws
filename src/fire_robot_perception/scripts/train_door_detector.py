#!/usr/bin/env python3
"""
train_door_detector.py

YOLOv8로 문(Door) 탐지 모델 학습.
  - 데이터셋: prepare_dataset.py로 준비한 OpenImages v7 Door (YOLO 형식)
  - 검증셋: dataset.yaml의 val 분할로 자동 평가
  - 최종 모델: runs/detect/door_detector/weights/best.pt

사용법:
  # 1단계: 데이터셋 준비
  python3 prepare_dataset.py --output_dir ./datasets/door_detection

  # 2단계: 학습
  python3 train_door_detector.py --dataset ./datasets/door_detection/dataset.yaml

  # 학습 후 ROS2 노드에서 사용:
  ros2 run fire_robot_perception door_detection_node \\
    --ros-args -p model_path:=./runs/detect/door_detector/weights/best.pt
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   help='dataset.yaml 경로')
    p.add_argument('--model', default='yolov8s.pt',
                   choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
                   help='기본 모델 크기 (n=nano, s=small, m=medium)')
    p.add_argument('--epochs',   type=int,   default=50)
    p.add_argument('--imgsz',    type=int,   default=640)
    p.add_argument('--batch',    type=int,   default=16)
    p.add_argument('--workers',  type=int,   default=4)
    p.add_argument('--device',   default='',
                   help='학습 디바이스 (빈 문자열=자동, "0"=GPU0, "cpu"=CPU)')
    p.add_argument('--project',  default='runs/detect')
    p.add_argument('--name',     default='door_detector')
    p.add_argument('--resume',   action='store_true',
                   help='이전 학습 이어서 진행')
    p.add_argument('--patience', type=int, default=15,
                   help='Early stopping patience (val/mAP50 기준)')
    return p.parse_args()


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print('ERROR: ultralytics가 설치되지 않았습니다.')
        print('설치: pip install ultralytics')
        return

    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f'ERROR: 데이터셋 파일을 찾을 수 없습니다: {dataset_path}')
        print('먼저 prepare_dataset.py를 실행하세요.')
        return

    # ── 모델 초기화 ─────────────────────────���─────────────
    if args.resume:
        last_ckpt = Path(args.project) / args.name / 'weights' / 'last.pt'
        if last_ckpt.exists():
            print(f'이전 체크포인트에서 재개: {last_ckpt}')
            model = YOLO(str(last_ckpt))
        else:
            print('이전 체크포인트 없음. 새로 시작합니다.')
            model = YOLO(args.model)
    else:
        print(f'기본 모델 로드: {args.model}')
        model = YOLO(args.model)

    # ── 학습 ─────────────────────────────────────────────
    print(f'\n학습 시작:')
    print(f'  데이터셋: {dataset_path}')
    print(f'  에폭: {args.epochs}  배치: {args.batch}  이미지: {args.imgsz}')
    print(f'  저장 경로: {args.project}/{args.name}')

    results = model.train(
        data      = str(dataset_path),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        device    = args.device if args.device else None,
        project   = args.project,
        name      = args.name,
        patience  = args.patience,
        resume    = args.resume,
        # ── 데이터 증강 (문 색상 변화에 강건하게) ──────────
        hsv_h     = 0.015,  # 색조 변동 최소화 (색상 분류 중요)
        hsv_s     = 0.4,    # 채도 변동
        hsv_v     = 0.4,    # 명도 변동
        degrees   = 5.0,    # 회전
        translate = 0.1,
        scale     = 0.4,
        flipud    = 0.0,    # 상하 뒤집기 없음 (문은 방향 있음)
        fliplr    = 0.3,    # 좌우 뒤집기
        mosaic    = 0.8,    # Mosaic 증강
        mixup     = 0.1,
        # ── 학습 파라미터 ───────────────────────��───────
        lr0       = 0.01,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        save       = True,
        save_period = 10,   # 10 에폭마다 체크포인트
        val        = True,  # 매 에폭 검증
        plots      = True,  # 학습 곡선 저장
    )

    # ── 검증 결과 출력 ────────────────────────────────────
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    print('\n' + '='*60)
    print('학습 완료!')
    print(f'  최적 모델: {best_model_path}')

    print('\n검증셋 최종 평가:')
    val_results = model.val(data=str(dataset_path), split='val')
    print(f'  mAP50:    {val_results.box.map50:.4f}')
    print(f'  mAP50-95: {val_results.box.map:.4f}')
    print(f'  Precision:{val_results.box.mp:.4f}')
    print(f'  Recall:   {val_results.box.mr:.4f}')
    print('='*60)

    print(f'\nROS2 노드에서 사용:')
    print(f'  ros2 run fire_robot_perception door_detection_node \\')
    print(f'    --ros-args -p model_path:={best_model_path.resolve()}')


if __name__ == '__main__':
    main()
