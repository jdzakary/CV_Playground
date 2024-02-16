from __future__ import annotations
from typing import TYPE_CHECKING

import cv2

from app.streaming.processing import Operation

if TYPE_CHECKING:
    import numpy as np


class SimpleDifference(Operation):
    name = 'Simple Difference'
    description = 'Subtracts current frame from previous frame'

    def __init__(self):
        super().__init__()
        self.__previous: np.ndarray

    @property
    def previous(self):
        return self.__previous

    @previous.setter
    def previous(self, value: np.ndarray):
        frame = value.copy()
        # noinspection PyAttributeOutsideInit
        self.__previous = frame

    def execute(self, frame: np.ndarray) -> np.ndarray:
        try:
            result = cv2.absdiff(frame, self.previous)
        except AttributeError:
            result = frame
        finally:
            self.previous = frame
            return result
