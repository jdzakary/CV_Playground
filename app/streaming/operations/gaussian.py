import cv2
import numpy as np

from app.streaming.processing import Operation, Slider


class GaussianBlur(Operation):
    name = 'Gaussian Blur'
    description = 'Blurs the frame using a gaussian filter'

    def __init__(self):
        super().__init__()
        self.__sigma = Slider(
            minimum=1,
            maximum=10,
            step=1,
            name='Sigma'
        )
        self.params.append(self.sigma)

    @property
    def sigma(self) -> Slider:
        return self.__sigma

    def execute(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (0, 0), self.sigma.number)
