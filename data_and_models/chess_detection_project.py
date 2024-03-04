import json
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

DATASET_FOLDER = Path(r'D:\Datasets\Computer Vision\Chesspiece Detection Modified')


def fix_class_labels(file_name: str | os.PathLike[str]) -> None:
    with open(file_name, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for i in lines:
        parts = i.split(' ', maxsplit=1)
        parts[0] = '1' if int(parts[0]) > 5 else '0'
        new_lines.append(' '.join(parts))
    with open(file_name, 'w') as f:
        f.writelines(new_lines)


def adjust_labels() -> None:
    total = len(os.listdir(DATASET_FOLDER / 'test/labels'))
    total += len(os.listdir(DATASET_FOLDER / 'train/labels'))
    total += len(os.listdir(DATASET_FOLDER / 'valid/labels'))
    with tqdm(total=total) as pbar:
        for folder in ['train', 'test', 'valid']:
            for file in os.listdir(DATASET_FOLDER / folder / 'labels'):
                fix_class_labels(DATASET_FOLDER / folder / 'labels' / file)
                pbar.update(1)


def train() -> None:
    model = YOLO('yolov8n.pt')
    model.cuda(0)
    results: DetMetrics = model.train(
        data=(DATASET_FOLDER / 'data.yaml').absolute(),
        epochs=200,
        imgsz=416,
        device=0,
        plots=True,
        batch=48,
    )
    print(results.results_dict)


if __name__ == '__main__':
    train()
