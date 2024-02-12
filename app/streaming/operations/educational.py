import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace

from app.streaming.processing import Operation, SingleSelect, Slider


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
        match self.__options.status:
            case 'Magnitude':
                result = np.hypot(row, column)
            case 'Row':
                result = row
            case 'Column':
                result = column
            case 'Angle':
                result = np.arctan2(column, row)
            case _:
                raise ValueError(f'Invalid Options: {self.__options.status}')

        result *= 255.0 / result.max()
        result = result.astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


class ECE4354EdgeDetectors(Operation):
    name = 'Learning - Edge Detectors'
    description = 'Homework assignment for ECE 4354'

    def __init__(self):
        super().__init__()
        self.__options = SingleSelect(
            labels=[
                'Canny', 'Kirsch', 'Laplacian',
                'LoG', 'Roberts', 'Sobel',
            ],
            name='Detector Type'
        )
        self.__logSigma = Slider(1, 8, name='LoG Sigma')
        self.params.append(self.__options)
        self.params.append(self.__logSigma)
        self.__kirsch = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        ]

    @property
    def options(self) -> SingleSelect:
        return self.__options

    def execute(self, frame: np.ndarray) -> np.ndarray:
        match self.__options.status:
            case 'Canny':
                return frame
            case 'Kirsch':
                h, w, d = frame.shape
                result = np.zeros((h, w, d, 8), dtype=np.uint8)
                for i, k in enumerate(self.__kirsch):
                    result[:, :, :, i] = cv2.filter2D(frame, -1, k)
                return result.max(axis=3)
            case 'Laplacian':
                k = np.array([
                    [-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]
                ])
                return cv2.filter2D(frame, -1, k)
            case 'LoG':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                return gaussian_laplace(gray, sigma=self.__logSigma.number).astype(np.uint8)
            case 'Roberts':
                kx = np.array([[1, 0], [0, -1]])
                ky = np.array([[0, 1], [-1, 0]])
                return cv2.filter2D(frame, -1, kx) + cv2.filter2D(frame, -1, ky)
            case 'Sobel':
                kx = np.array([
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]
                ])
                ky = np.array([
                    [-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]
                ])
                return cv2.filter2D(frame, -1, kx) + cv2.filter2D(frame, -1, ky)
            case _:
                raise ValueError(f'Invalid Options: {self.__options.status}')
