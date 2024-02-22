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


class HarrisCorners(Operation):
    name = "Harris Corner"
    description = "Detect Corners"

    def __init__(self):
        super().__init__()
        self.__threshold = NewSlider(
            name="Corner Threshold",
            minimum=0.01,
            maximum=0.10,
            step=0.01,
            label_precision=2,
            default=0.04
        )
        self.__thickness = SingleSelect(
            name="Marker Thickness",
            labels=['Filled', 'Outline Thin', 'Outline Thick']
        )
        self.__radius = NewSlider(
            name="Marker Radius",
            minimum=1,
            maximum=8,
            default=4,
        )
        self.params.append(self.__threshold)
        self.params.append(self.__thickness)
        self.params.append(self.__radius)

    def __convert_thickness(self) -> int:
        match self.__thickness.status:
            case 'Filled':
                return -1
            case 'Outline Thin':
                return 1
            case 'Outline Thick':
                return 2

    def execute(self, frame: np.ndarray) -> np.ndarray:
        gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        strength = cv2.cornerHarris(
            gray,
            2,
            3,
            0.04
        )
        _, thresh = cv2.threshold(
            strength,
            self.__threshold.number * strength.max(),
            255,
            cv2.THRESH_BINARY
        )
        thresh = np.uint8(thresh)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        centroids = np.float32(centroids)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, centroids, (5, 5), (-1, -1), criteria)

        for i in range(1, corners.shape[0]):
            cv2.circle(
                frame,
                (int(corners[i, 0]), int(corners[i, 1])),
                int(self.__radius.number),
                (0, 0, 225),
                int(self.__convert_thickness())
            )
        return frame
