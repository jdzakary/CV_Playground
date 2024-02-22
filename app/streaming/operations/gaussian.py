from __future__ import annotations
from typing import TYPE_CHECKING

import cv2

from app.streaming.processing import Operation, NewSlider

if TYPE_CHECKING:
    import numpy as np


class GaussianBlur(Operation):
    name = 'Gaussian Blur'
    description = 'Blurs the frame using a gaussian filter'

    def __init__(self):
        super().__init__()
        self.__sigma = NewSlider(
            minimum=1,
            maximum=10,
            step=0.5,
            name='Sigma',
            default=1,
            label_precision=1,
        )
        self.params.append(self.__sigma)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (0, 0), self.__sigma.number)
