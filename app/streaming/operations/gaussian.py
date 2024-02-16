from __future__ import annotations
from typing import TYPE_CHECKING

import cv2

from app.streaming.processing import Operation, Slider

if TYPE_CHECKING:
    import numpy as np


class GaussianBlur(Operation):
    name = 'Gaussian Blur'
    description = 'Blurs the frame using a gaussian filter'

    def __init__(self):
        super().__init__()
        self.__sigma = Slider(
            minimum=1,
            maximum=10,
            step=1,
            name='Sigma',
            default=1,
            divisor=2,
        )
        self.params.append(self.sigma)

    @property
    def sigma(self) -> Slider:
        return self.__sigma

    def execute(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (0, 0), self.sigma.number)
