import cv2
import numpy as np

from app.streaming.processing import Operation, NewSlider, SingleSelect


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
