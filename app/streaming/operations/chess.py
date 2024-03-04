from __future__ import annotations
from typing import TYPE_CHECKING

import cv2
import numpy as np
from ultralytics import YOLO

from app.streaming.processing import Operation, NewSlider, IntegerEntry

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class ChessDetection1(Operation):
    name = "Chess Piece Detection"
    description = "Draw Bounding Boxes around chess pieces"
    InitializeInSeparateThread = True

    def __init__(self):
        super().__init__()
        self.__model = YOLO('data_and_models/runs/detect/train2/weights/best.pt')
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


class BoardDetection(Operation):
    name = "Board Detection"
    description = "Compute game board grid"

    def __init__(self):
        super().__init__()
        self.__max_corners = IntegerEntry(
            name='Max Corners',
            min_value=1,
            max_value=1000,
            step=5,
            default=25
        )
        self.__test_point = IntegerEntry(
            name='Test Point',
            min_value=1,
            max_value=25,
            step=1,
            default=1
        )
        self.params.append(self.__max_corners)
        self.params.append(self.__test_point)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        idx = self.__test_point.number
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray,
            self.__max_corners.number,
            0.01,
            20,
        )
        vectors = corners - np.swapaxes(corners, 0, 1)
        angles = self.__compute_angles(vectors)

        selected = angles[idx, idx, :, :]
        ideal = np.abs(selected - 90) < 10

        corners = np.intp(corners)
        for i, data in enumerate(corners):
            x, y = data.ravel()
            if i == idx:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(
                frame,
                (x, y),
                2,
                color,
                -1,
            )
        return frame

    @staticmethod
    def __compute_angles(vectors: np.ndarray) -> np.ndarray:
        size = vectors.shape[0]
        mag = np.sqrt(np.sum(np.square(vectors), axis=2))
        mag = mag * mag.T
        dot = np.outer(vectors[:, :, 0], vectors[:, :, 0]) + np.outer(vectors[:, :, 1], vectors[:, :, 1])
        dot = dot.reshape((size, size, size, size))
        result = np.arccos(dot / mag)
        result = np.nan_to_num(result)
        return np.rad2deg(result)
