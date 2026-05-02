#!/usr/bin/env python3
"""
prepare_dataset.py

OpenImages v7 "Door" + Roboflow door 데이터셋을 병합하여 YOLOv8 학습 형식으로 저장.

사용법:
  # OpenImages만 사용
  python3 prepare_dataset.py --output_dir ./datasets/door_detection

  # Roboflow 추가 병합 (무료 API 키 필요: https://app.roboflow.com → Settings → API)
  python3 prepare_dataset.py --output_dir ./datasets/door_detection \\
      --roboflow_key YOUR_API_KEY

  # Roboflow 커스텀 데이터셋 지정
  python3 prepare_dataset.py --output_dir ./datasets/door_detection \\
      --roboflow_key YOUR_API_KEY \\
      --roboflow_workspace my-workspace \\
      --roboflow_project door-detection \\
      --roboflow_version 3

의존성:
  pip install fiftyone roboflow Pillow tqdm
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    _HAS_FO = True
except ImportError:
    _HAS_FO = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from roboflow import Roboflow
    _HAS_RF = True
except ImportError:
    _HAS_RF = False


# Roboflow 기본 데이터셋: Roboflow-100 벤치마크의 door-detection
# https://universe.roboflow.com/roboflow-100/door-detection-3d8km
_RF_DEFAULT_WORKSPACE = 'roboflow-100'
_RF_DEFAULT_PROJECT   = 'door-detection-3d8km'
_RF_DEFAULT_VERSION   = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='./datasets/door_detection')
    p.add_argument('--max_samples_train', type=int, default=5000)
    p.add_argument('--max_samples_val',   type=int, default=500)
    p.add_argument('--classes', nargs='+', default=['Door'])
    p.add_argument('--roboflow_key',       default='',
                   help='Roboflow API 키 (roboflow.com → Settings → API Key)')
    p.add_argument('--roboflow_workspace', default=_RF_DEFAULT_WORKSPACE)
    p.add_argument('--roboflow_project',   default=_RF_DEFAULT_PROJECT)
    p.add_argument('--roboflow_version',   type=int, default=_RF_DEFAULT_VERSION)
    return p.parse_args()


# ── OpenImages ────────────────────────────────────────────────────────────────

def download_openimages(split: str, max_samples: int, classes: list) -> 'fo.Dataset':
    print(f'\n[OpenImages/{split}] "{classes}" 다운로드 중 (최대 {max_samples}장)...')
    dataset = foz.load_zoo_dataset(
        'open-images-v7',
        split=split,
        label_types=['detections'],
        classes=classes,
        max_samples=max_samples,
        shuffle=True,
    )
    print(f'  완료: {len(dataset)}장')
    return dataset


def convert_to_yolo(dataset: 'fo.Dataset', out_img_dir: Path,
                    out_lbl_dir: Path, class_names: list):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    class_map = {name.lower(): idx for idx, name in enumerate(class_names)}
    skipped = 0

    for sample in tqdm(dataset, desc='YOLO 변환'):
        src = Path(sample.filepath)
        if not src.exists():
            skipped += 1
            continue
        shutil.copy2(src, out_img_dir / src.name)

        if _HAS_PIL:
            with Image.open(src) as img:
                iw, ih = img.size
        else:
            iw, ih = 640, 480

        lines = []
        if sample.ground_truth is not None:
            for det in sample.ground_truth.detections:
                cls_id = class_map.get(det.label.lower(), -1)
                if cls_id < 0:
                    continue
                x1, y1, bw, bh = det.bounding_box
                lines.append(
                    f'{cls_id} {x1+bw/2:.6f} {y1+bh/2:.6f} {bw:.6f} {bh:.6f}')

        (out_lbl_dir / (src.stem + '.txt')).write_text('\n'.join(lines))

    if skipped:
        print(f'  경고: {skipped}개 파일 없음 (스킵)')


# ── Roboflow ──────────────────────────────────────────────────────────────────

def download_roboflow(api_key: str, workspace: str, project: str,
                      version: int) -> Path:
    """Roboflow에서 YOLOv8 형식으로 다운로드 → 임시 디렉토리 반환"""
    print(f'\n[Roboflow] {workspace}/{project} v{version} 다운로드 중...')
    rf = Roboflow(api_key=api_key)
    tmp = Path(tempfile.mkdtemp(prefix='roboflow_'))
    try:
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download('yolov8', location=str(tmp))
    except Exception as e:
        print(f'  ERROR: {e}')
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    print(f'  다운로드 완료: {tmp}')
    return tmp


def merge_roboflow(rf_dir: Path, out_dir: Path):
    """Roboflow 디렉토리(train/valid) → 메인 데이터셋(train/val) 에 병합"""
    # Roboflow 출력: train/images, train/labels, valid/images, valid/labels
    mapping = [
        (rf_dir / 'train' / 'images', out_dir / 'images' / 'train'),
        (rf_dir / 'train' / 'labels', out_dir / 'labels' / 'train'),
        (rf_dir / 'valid' / 'images', out_dir / 'images' / 'val'),
        (rf_dir / 'valid' / 'labels', out_dir / 'labels' / 'val'),
    ]
    total = 0
    for src_dir, dst_dir in mapping:
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            dst = dst_dir / f.name
            # 중복 파일명 충돌 방지
            if dst.exists():
                dst = dst_dir / (f'rf_' + f.name)
            shutil.copy2(f, dst)
            total += 1
    print(f'  Roboflow 병합: {total}개 파일 추가')


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def write_dataset_yaml(out_dir: Path, class_names: list):
    content = (
        f'path: {out_dir.resolve()}\n'
        f'train: images/train\n'
        f'val:   images/val\n\n'
        f'nc: {len(class_names)}\n'
        f'names: {class_names}\n'
    )
    (out_dir / 'dataset.yaml').write_text(content)
    print(f'\ndataset.yaml 저장: {out_dir / "dataset.yaml"}')


def _count_images(d: Path) -> int:
    return len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png')))


def _split_exists(img_dir: Path, min_images: int = 10) -> bool:
    return img_dir.exists() and _count_images(img_dir) >= min_images


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not _HAS_FO:
        print('ERROR: fiftyone 미설치.  pip install fiftyone')
        return

    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    class_names = args.classes

    # ── 1. OpenImages v7 ─────────────────────────────────
    train_img = outdir / 'images' / 'train'
    if _split_exists(train_img):
        print(f'[OpenImages/train] 이미 존재 ({_count_images(train_img)}장) — 스킵')
    else:
        ds = download_openimages('train', args.max_samples_train, class_names)
        convert_to_yolo(ds, train_img, outdir / 'labels' / 'train', class_names)
        ds.delete()

    val_img = outdir / 'images' / 'val'
    if _split_exists(val_img):
        print(f'[OpenImages/val] 이미 존재 ({_count_images(val_img)}장) — 스킵')
    else:
        ds = download_openimages('validation', args.max_samples_val, class_names)
        convert_to_yolo(ds, val_img, outdir / 'labels' / 'val', class_names)
        ds.delete()

    # ── 2. Roboflow (선택) ───────────────────────────────
    if args.roboflow_key:
        if not _HAS_RF:
            print('WARNING: roboflow 미설치.  pip install roboflow  — 스킵')
        else:
            rf_tmp = download_roboflow(
                args.roboflow_key,
                args.roboflow_workspace,
                args.roboflow_project,
                args.roboflow_version,
            )
            if rf_tmp is not None:
                merge_roboflow(rf_tmp, outdir)
                shutil.rmtree(rf_tmp, ignore_errors=True)
    else:
        print('\n[Roboflow] API 키 없음 — OpenImages만 사용')
        print('  Roboflow 추가: --roboflow_key YOUR_KEY')
        print('  무료 키 발급: https://app.roboflow.com → Settings → API Key')

    # ── 3. dataset.yaml 갱신 ────────────────────────────
    write_dataset_yaml(outdir, class_names)

    n_train = _count_images(outdir / 'images' / 'train')
    n_val   = _count_images(outdir / 'images' / 'val')
    print(f'\n데이터셋 준비 완료:')
    print(f'  학습: {n_train}장  |  검증: {n_val}장')
    print(f'  경로: {outdir.resolve()}')
    print(f'\n학습 실행:')
    print(f'  python3 train_door_detector.py --dataset {outdir / "dataset.yaml"}')


if __name__ == '__main__':
    main()
