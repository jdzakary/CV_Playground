import cv2
import numpy as np

from app.video.processing import Operation, Slider


class CannyEdges(Operation):
    name = 'Canny Edge Detector'
    description = 'The canny edge detector implemented by opencv'

    def __init__(self):
        super().__init__()
        self.__thresh_1 = Slider(50, 200, 10)
        self.__thresh_2 = Slider(200, 300, 10)
        self.thresh_1.slider.valueChanged.connect(self.adjust_min)
        self.thresh_2.slider.valueChanged.connect(self.adjust_max)
        self.params.append(self.__thresh_1)
        self.params.append(self.__thresh_2)

    @property
    def thresh_1(self) -> Slider:
        return self.__thresh_1

    @property
    def thresh_2(self) -> Slider:
        return self.__thresh_2

    def adjust_min(self, value: int) -> None:
        self.thresh_2.minimum = value

    def adjust_max(self, value: int) -> None:
        self.thresh_1.maximum = value

    def execute(self, frame: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(frame, self.thresh_1.number, self.thresh_2.number)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return cv2.add(frame, edges)
