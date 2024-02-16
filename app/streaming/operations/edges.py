from __future__ import annotations
from typing import TYPE_CHECKING

import cv2

from app.streaming.processing import Operation, Slider, Boolean

if TYPE_CHECKING:
    import numpy as np


class CannyEdges(Operation):
    name = 'Canny Edge Detector'
    description = 'The canny edge detector implemented by opencv'

    def __init__(self):
        super().__init__()
        self.__thresh_1 = Slider(
            minimum=50,
            maximum=200,
            step=10,
            name='Lower Threshold',
            default=100,
        )
        self.__thresh_2 = Slider(
            minimum=100,
            maximum=300,
            step=10,
            name='Upper Threshold',
            default=200
        )
        self.__show_frames = Boolean(
            label_true='Show Frames',
            label_false='Only Edges',
            name='Display Type'
        )
        self.thresh_1.slider.valueChanged.connect(self.adjust_min)
        self.thresh_2.slider.valueChanged.connect(self.adjust_max)
        self.params.append(self.__thresh_1)
        self.params.append(self.__thresh_2)
        self.params.append(self.show_frames)

    @property
    def thresh_1(self) -> Slider:
        return self.__thresh_1

    @property
    def thresh_2(self) -> Slider:
        return self.__thresh_2

    @property
    def show_frames(self) -> Boolean:
        return self.__show_frames

    def adjust_min(self, value: int) -> None:
        self.thresh_2.minimum = value

    def adjust_max(self, value: int) -> None:
        self.thresh_1.maximum = value

    def execute(self, frame: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(frame, self.thresh_1.number, self.thresh_2.number)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        if self.show_frames.status:
            return cv2.add(frame, edges)
        else:
            return edges
