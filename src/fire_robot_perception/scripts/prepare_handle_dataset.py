#!/usr/bin/env python3
"""Build a one-class YOLO door-handle dataset from Open Images V7.

The script deliberately streams the large Open Images CSV files so they are
never loaded into memory.  It expects metadata downloaded by FiftyOne under
``~/fiftyone/open-images-v7`` and writes only the selected images into the
workspace dataset directory.
"""

import argparse
import csv
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HANDLE_LABEL_ID = '/m/03c7gz'
IMAGE_SUFFIX = '.jpg'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open-images-dir',
                        default='~/fiftyone/open-images-v7')
    parser.add_argument('--output-dir',
                        default='datasets/handle_detection')
    parser.add_argument('--train-count', type=int, default=400)
    parser.add_argument('--val-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=12)
    return parser.parse_args()


def collect_handle_boxes(annotation_csv: Path) -> dict[str, list[list[float]]]:
    boxes: dict[str, list[list[float]]] = {}
    with annotation_csv.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['LabelName'] != HANDLE_LABEL_ID:
                continue
            xmin = float(row['XMin'])
            xmax = float(row['XMax'])
            ymin = float(row['YMin'])
            ymax = float(row['YMax'])
            boxes.setdefault(row['ImageID'], []).append([
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                xmax - xmin,
                ymax - ymin,
            ])
    return boxes


def collect_image_metadata(image_csv: Path,
                           wanted_ids: set[str]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with image_csv.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            image_id = row['ImageID']
            if image_id in wanted_ids:
                metadata[image_id] = row
                if len(metadata) == len(wanted_ids):
                    break
    return metadata


def download_one(url: str, destination: Path) -> tuple[bool, str]:
    try:
        request = urllib.request.Request(
            url, headers={'User-Agent': 'fire-robot-handle-dataset/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            destination.write_bytes(response.read())
        return True, ''
    except Exception as exc:
        return False, str(exc)


def prepare_split(source_split: str, output_split: str, count: int, source_root: Path,
                  output_root: Path, rng: random.Random,
                  workers: int) -> tuple[int, int]:
    metadata_root = source_root / source_split / 'metadata'
    labels_root = source_root / source_split / 'labels'
    annotation_csv = labels_root / 'detections.csv'
    image_csv = metadata_root / 'image_ids.csv'
    if not annotation_csv.exists() or not image_csv.exists():
        raise FileNotFoundError(
            f'Missing Open Images {source_split} metadata. Expected '
            f'{annotation_csv} and {image_csv}')

    boxes_by_id = collect_handle_boxes(annotation_csv)
    image_ids = sorted(boxes_by_id)
    rng.shuffle(image_ids)
    candidates = image_ids[:max(count * 2, count + 20)]
    metadata = collect_image_metadata(image_csv, set(candidates))

    image_dir = output_root / 'images' / output_split
    label_dir = output_root / 'labels' / output_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    selected = [image_id for image_id in candidates if image_id in metadata]
    downloaded: list[tuple[str, dict[str, str]]] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for image_id in selected:
            if len(futures) >= count * 2:
                break
            row = metadata[image_id]
            # Use the canonical Open Images public bucket; the Flickr URLs in
            # image_ids.csv frequently expire or return HTTP 502.
            url = (
                'https://open-images-dataset.s3.amazonaws.com/'
                f'{source_split}/{image_id}.jpg'
            )
            destination = image_dir / f'{image_id}{IMAGE_SUFFIX}'
            futures[pool.submit(download_one, url, destination)] = (
                image_id, row, destination)

        for future in as_completed(futures):
            image_id, row, destination = futures[future]
            ok, error = future.result()
            if not ok:
                failures += 1
                destination.unlink(missing_ok=True)
                print(f'WARN download failed {image_id}: {error}')
                continue
            downloaded.append((image_id, row))
            if len(downloaded) >= count:
                break

    # Remove successful downloads beyond the requested deterministic count.
    downloaded.sort(key=lambda item: selected.index(item[0]))
    keep = downloaded[:count]
    keep_ids = {image_id for image_id, _ in keep}
    for path in image_dir.glob(f'*{IMAGE_SUFFIX}'):
        if path.stem not in keep_ids:
            path.unlink()

    for image_id, _ in keep:
        lines = [
            '0 ' + ' '.join(f'{value:.8f}' for value in box)
            for box in boxes_by_id[image_id]
        ]
        (label_dir / f'{image_id}.txt').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')

    attribution = output_root / f'open_images_{output_split}_attribution.csv'
    with attribution.open('w', newline='', encoding='utf-8') as stream:
        fields = ['ImageID', 'OriginalURL', 'OriginalLandingURL',
                  'License', 'Author', 'Title']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_id, row in keep:
            writer.writerow({field: row.get(field, '') for field in fields})

    return len(keep), failures


def main():
    args = parse_args()
    source_root = Path(args.open_images_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    rng = random.Random(args.seed)

    train_count, train_failures = prepare_split(
        'train', 'train', args.train_count, source_root, output_root,
        rng, args.workers)
    val_count, val_failures = prepare_split(
        'test', 'val', args.val_count, source_root, output_root,
        rng, args.workers)

    dataset_yaml = output_root / 'dataset.yaml'
    dataset_yaml.write_text(
        f'path: {output_root}\n'
        'train: images/train\n'
        'val: images/val\n'
        'names:\n'
        '  0: lever_handle\n',
        encoding='utf-8')
    print(f'train images: {train_count} (download failures: {train_failures})')
    print(f'validation images: {val_count} (download failures: {val_failures})')
    print(f'dataset: {dataset_yaml}')
    if train_count < args.train_count or val_count < args.val_count:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
