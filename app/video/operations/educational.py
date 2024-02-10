import cv2
import numpy as np

from app.video.processing import Operation, SingleSelect


class ECE4354Gradients(Operation):
    name = 'Learning - Gradients'
    description = 'Homework assignment for ECE 4354'

    def __init__(self):
        super().__init__()
        self.__options = SingleSelect(
            labels=['Magnitude', 'Row', 'Column', 'Angle'],
            name='Gradient Type'
        )
        self.params.append(self.__options)

    @property
    def options(self) -> SingleSelect:
        return self.__options

    def execute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row, column = np.gradient(gray)
        if self.__options.status == 'Magnitude':
            result = np.hypot(row, column)
        elif self.__options.status == 'Row':
            result = row
        elif self.__options.status == 'Column':
            result = column
        elif self.__options.status == 'Angle':
            result = np.arctan2(row, column)
        else:
            raise Exception('Invalid Options')

        result *= 255.0 / result.max()
        result = result.astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
