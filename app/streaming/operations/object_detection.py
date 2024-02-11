import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from app.video.processing import Operation, Slider


class Yolo(Operation):
    name = "YOLO Object Detection"
    description = "Detects Objects using You Only Look Once"

    def __init__(self):
        super().__init__()
        self.__model = YOLO('yolov8n.pt')
        self.__model.cuda(0)
        self.__conf = Slider(1, 100, 1)
        self.params.append(self.__conf)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        h, w, d = frame.shape
        # noinspection PyTypeChecker
        results: Results = self.__model(
            source=frame,
            device='0',
            imgsz=(h, w),
            verbose=False,
            conf=self.__conf.number / 100
        )
        return results[0].plot()