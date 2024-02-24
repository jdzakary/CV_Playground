from __future__ import annotations

import cv2
import numpy as np

from app.streaming.processing import Operation, Boolean, NewSlider, SingleSelect


class CannyEdges(Operation):
    name = 'Canny Edge Detector'
    description = 'The canny edge detector implemented by opencv'

    def __init__(self):
        super().__init__()
        self.__thresh_1 = NewSlider(
            minimum=50,
            maximum=200,
            step=5,
            name='Lower Threshold',
            default=100,
        )
        self.__thresh_2 = NewSlider(
            minimum=100,
            maximum=300,
            step=5,
            name='Upper Threshold',
            default=200
        )
        self.__show_frames = Boolean(
            label_true='Show Frames',
            label_false='Only Edges',
            name='Display Type'
        )
        self.params.append(self.__thresh_1)
        self.params.append(self.__thresh_2)
        self.params.append(self.__show_frames)
        self.child_functions.append(self._connect_signals)

    def __adjust_min(self) -> None:
        self.__thresh_2.minimum = self.__thresh_1.number

    def __adjust_max(self) -> None:
        self.__thresh_1.maximum = self.__thresh_2.number

    def _connect_signals(self) -> None:
        self.__thresh_1.slider.valueChanged.connect(self.__adjust_min)
        self.__thresh_2.slider.valueChanged.connect(self.__adjust_max)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(frame, self.__thresh_1.number, self.__thresh_2.number)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        if self.__show_frames.status:
            return cv2.add(frame, edges)
        else:
            return edges
