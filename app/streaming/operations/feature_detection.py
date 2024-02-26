import math

import cv2
import numpy as np

from app.streaming.processing import Operation, NewSlider, SingleSelect, Boolean, IntegerEntry


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
        self.__nms = Boolean(
            name='Perform NMS',
            label_true='Yes',
            label_false='No'
        )
        self.__nms_neighborhood_radius = NewSlider(
            name="NMS Neighborhood Radius",
            minimum=1,
            maximum=10,
            step=1,
            default=3,
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
        self.params.append(self.__nms)
        self.params.append(self.__nms_neighborhood_radius)
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
            cv2.THRESH_TOZERO
        )

        if self.__nms.status:
            dilated = cv2.dilate(
                thresh,
                None,
                iterations=int(self.__nms_neighborhood_radius.number)
            )
            thresh = 255 * (thresh == dilated)

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
        if self.__nms.status:
            cv2.putText(
                img=frame,
                text=f'NMS On - Radius {self.__nms_neighborhood_radius.number}',
                org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return frame


class SIFT(Operation):
    name = 'SIFT'
    description = 'Scale Invariant Feature Transform'

    def __init__(self):
        super().__init__()
        self.__sift = cv2.SIFT.create()

    def execute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = self.__sift.detect(gray)
        frame = cv2.drawKeypoints(
            frame,
            kp,
            frame,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return frame


class GFTT(Operation):
    name = 'GFTT'
    description = 'Good Features to Track'

    def __init__(self):
        super().__init__()
        self.__max_corners = IntegerEntry(
            name='Max Corners',
            min_value=1,
            max_value=1000,
            step=5,
            default=25
        )
        self.__quality_level = NewSlider(
            name="Quality Level",
            minimum=0.01,
            maximum=0.10,
            step=0.01,
            label_precision=2,
            default=0.04
        )
        self.__min_distance = IntegerEntry(
            name='Min Corner Distance',
            min_value=1,
            step=5,
            default=10
        )
        self.params.append(self.__max_corners)
        self.params.append(self.__quality_level)
        self.params.append(self.__min_distance)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray,
            self.__max_corners.number,
            self.__quality_level.number,
            self.__min_distance.number,
        )

        corners = np.intp(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 0, 255),
                -1,
            )
        return frame


class ORB(Operation):
    name = 'ORB'
    description = 'ORB Feature Detector'

    def __init__(self):
        super().__init__()
        self.__nms = Boolean(
            name='Perform NMS',
            label_true='Yes',
            label_false='No'
        )
        self.__orb = cv2.ORB.create()
        self.params.append(self.__nms)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = self.__orb.detect(gray)
        kp = sorted(kp, key=lambda x: x.response, reverse=True)

        if self.__nms.status:
            kp = self.ssc(kp, 10, 0.04, frame.shape[1], frame.shape[0])

        frame = cv2.drawKeypoints(
            frame,
            kp,
            frame,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        if self.__nms.status:
            cv2.putText(
                img=frame,
                text=f'NMS On',
                org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def ssc(key_points, num_ret_points, tolerance, cols, rows):
        """
        This is the implementation of the [paper](
        https://www.researchgate.net/publication/323388062_
        Efficient_adaptive_non-maximal_suppression_algorithms_for_homogeneous_spatial_keypoint_distribution)
        "Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution"
        that is published in Pattern Recognition Letters (PRL).
        Downloaded from https://github.com/BAILOOL/ANMS-Codes
        """
        exp1 = rows + cols + 2 * num_ret_points
        exp2 = 4 * cols + 4 * num_ret_points + 4 * rows * num_ret_points + rows * rows + cols * cols - \
               2 * rows * cols + 4 * rows * cols * num_ret_points
        exp3 = math.sqrt(exp2)
        exp4 = num_ret_points - 1

        sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
        sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

        high = sol1 if (sol1 > sol2) else sol2  # binary search range initialization with positive solution
        low = math.floor(math.sqrt(len(key_points) / num_ret_points))

        prev_width = -1
        selected_keypoints = []
        result_list = []
        result = []
        complete = False
        k = num_ret_points
        k_min = round(k - (k * tolerance))
        k_max = round(k + (k * tolerance))

        while not complete:
            width = low + (high - low) / 2
            if width == prev_width or low > high:  # needed to reassure the same radius is not repeated again
                result_list = result  # return the keypoints from the previous iteration
                break

            c = width / 2  # initializing Grid
            num_cell_cols = int(math.floor(cols / c))
            num_cell_rows = int(math.floor(rows / c))
            covered_vec = [[False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_cols + 1)]
            result = []

            for i in range(len(key_points)):
                row = int(math.floor(key_points[i].pt[1] / c))  # get position of the cell current point is located at
                col = int(math.floor(key_points[i].pt[0] / c))
                if not covered_vec[row][col]:  # if the cell is not covered
                    result.append(i)
                    # get range which current radius is covering
                    row_min = int((row - math.floor(width / c)) if ((row - math.floor(width / c)) >= 0) else 0)
                    row_max = int(
                        (row + math.floor(width / c)) if (
                            (row + math.floor(width / c)) <= num_cell_rows) else num_cell_rows)
                    col_min = int((col - math.floor(width / c)) if ((col - math.floor(width / c)) >= 0) else 0)
                    col_max = int(
                        (col + math.floor(width / c)) if (
                            (col + math.floor(width / c)) <= num_cell_cols) else num_cell_cols)
                    for rowToCov in range(row_min, row_max + 1):
                        for colToCov in range(col_min, col_max + 1):
                            if not covered_vec[rowToCov][colToCov]:
                                # cover cells within the square bounding box with width w
                                covered_vec[rowToCov][colToCov] = True

            if k_min <= len(result) <= k_max:  # solution found
                result_list = result
                complete = True
            elif len(result) < k_min:
                high = width - 1  # update binary search range
            else:
                low = width + 1
            prev_width = width

        for i in range(len(result_list)):
            selected_keypoints.append(key_points[result_list[i]])

        return selected_keypoints
