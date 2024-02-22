from __future__ import annotations
from typing import TYPE_CHECKING

from ultralytics import YOLO

from app.streaming.processing import Operation, NewSlider

if TYPE_CHECKING:
    import numpy as np
    from ultralytics.engine.results import Results


class YoloObjectSegmentation(Operation):
    name = "YOLO Object Segmentation"
    description = "Detailed Object Outlining"
    InitializeInSeparateThread = True

    def __init__(self):
        super().__init__()
        self.__model = YOLO('assets/yolov8n-seg.pt')
        self.__model.cuda(0)
        self.__conf = NewSlider(
            minimum=1,
            maximum=100,
            step=1,
            name='Confidence',
            default=80,
        )
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


class YoloObjectDetect(Operation):
    name = "YOLO Object Detection"
    description = "Detects Objects using You Only Look Once"
    InitializeInSeparateThread = True

    def __init__(self):
        super().__init__()
        self.__model = YOLO('assets/yolov8n.pt')
        self.__model.cuda(0)
        self.__conf = NewSlider(
            minimum=1,
            maximum=100,
            step=1,
            name='Confidence',
            default=80,
        )
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
