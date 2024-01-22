import cv2
import numpy as np

from app.video.processing import Operation, Slider


class GaussianBlur(Operation):
    name = 'Gaussian Blur'
    description = 'Blurs the frame using a gaussian filter'

    def __init__(self):
        super().__init__()
        self.__sigma = Slider(1, 10, 1)
        self.params.append(self.sigma)

    @property
    def sigma(self) -> Slider:
        return self.__sigma

    def execute(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (0, 0), self.sigma.number)
