import cv2
import numpy as np
from app.streaming.processing import Operation


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
        result = np.array([])
        try:
            result = cv2.absdiff(frame, self.previous)
        except AttributeError:
            result = frame
        finally:
            self.previous = frame
            return result
