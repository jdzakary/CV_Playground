from __future__ import annotations
from typing import TYPE_CHECKING
from enum import Enum

import cv2
import numpy as np
from ultralytics import YOLO

from app.streaming.processing import Operation, NewSlider, IntegerEntry, Boolean

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class ChessRole(Enum):
    King = 0
    Queen = 1
    Rook = 2
    Bishop = 3
    Knight = 4
    Pawn = 5


class ChessPiece:
    def __init__(self, white: bool, role: ChessRole, row: int, col: int, idx: int):
        if row < 0 or row > 7 or col < 0 or col > 7:
            raise Exception('Row and Column are bound between 0 and 7')
        self.__row = row
        self.__col = col
        self.__white = white
        self.__role = role
        self.__idx = idx

    @property
    def idx(self) -> int:
        return self.__idx

    @property
    def position(self) -> (int, int):
        return self.__row, self.__col

    @property
    def row(self) -> int:
        return self.__row

    @property
    def col(self) -> int:
        return self.__col

    @property
    def white(self) -> bool:
        return self.__white

    @property
    def role(self) -> ChessRole:
        return self.__role

    @position.setter
    def position(self, new_position: (int, int)) -> None:
        if new_position[0] < 0 or new_position[0] > 7 or new_position[1] < 0 or new_position[1] > 7:
            raise Exception('Row and col must be between 0 and 7')
        self.__row = new_position[0]
        self.__col = new_position[1]

    @row.setter
    def row(self, new_row: int) -> None:
        if new_row < 0 or new_row > 7:
            raise Exception('Row must be between 0 and 7')
        self.__row = new_row

    @col.setter
    def col(self, new_col: int) -> None:
        if new_col < 0 or new_col > 7:
            raise Exception('Col must be between 0 and 7')
        self.__col = new_col

    def promotion(self, target: ChessRole) -> None:
        if self.__role != ChessRole.Pawn:
            raise Exception('Only pawns can be promoted')
        if target == ChessRole.Pawn:
            raise Exception('Cannot promote a pawn to a pawn')
        if target == ChessRole.King:
            raise Exception('Cannot promote a pawn to king')
        self.__role = target


class ChessGame:
    class EmptySquare(Exception):
        """Raised when the chess square contains no pieces"""
        pass

    def __init__(self):
        self.__board = np.zeros((8, 8))
        self.__pieces = {
            1: ChessPiece(False, ChessRole.Rook, 0, 0, 1),
            2: ChessPiece(False, ChessRole.Knight, 0, 1, 2),
            3: ChessPiece(False, ChessRole.Bishop, 0, 2, 3),
            4: ChessPiece(False, ChessRole.Queen, 0, 3, 4),
            5: ChessPiece(False, ChessRole.King, 0, 4, 5),
            6: ChessPiece(False, ChessRole.Bishop, 0, 5, 6),
            7: ChessPiece(False, ChessRole.Knight, 0, 6, 7),
            8: ChessPiece(False, ChessRole.Rook, 0, 7, 8),
            9: ChessPiece(True, ChessRole.Rook, 7, 0, 9),
            10: ChessPiece(True, ChessRole.Knight, 7, 1, 10),
            11: ChessPiece(True, ChessRole.Bishop, 7, 2, 11),
            12: ChessPiece(True, ChessRole.Queen, 7, 3, 12),
            13: ChessPiece(True, ChessRole.King, 7, 4, 13),
            14: ChessPiece(True, ChessRole.Bishop, 7, 5, 14),
            15: ChessPiece(True, ChessRole.Knight, 7, 6, 15),
            16: ChessPiece(True, ChessRole.Rook, 7, 7, 16),
        }
        for i in range(8):
            self.__pieces[17 + i] = ChessPiece(False, ChessRole.Pawn, 1, i, 17 + i)
            self.__pieces[25 + i] = ChessPiece(False, ChessRole.Pawn, 6, i, 25 + i)
        self.__board[0, :] = np.arange(1, 9)
        self.__board[1, :] = np.arange(17, 25)
        self.__board[7, :] = np.arange(9, 17)
        self.__board[6, :] = np.arange(25, 33)

    def get_occupant(self, row: int, col: int) -> ChessPiece:
        if (idx := self.__board[row, col]) == 0:
            raise self.EmptySquare(f'No piece at position ({row}, {col})')
        return self.__pieces[idx]

    def is_occupied(self, row: int, col: int) -> bool:
        return self.__board[row, col] != 0

    def __remove_piece(self, row: int, col: int) -> None:
        occupant = self.get_occupant(row, col)
        self.__pieces.pop(occupant.idx)
        self.__board[row, col] = 0

    def move_piece(self, start: (int, int), end: (int, int)) -> None:
        occupant = self.get_occupant(*start)
        try:
            destination = self.get_occupant(*end)
        except self.EmptySquare:
            pass
        else:
            if destination.white == occupant.white:
                raise Exception('Cannot capture piece of same color')
            self.__remove_piece(*end)

        occupant.position = end
        self.__board[end[0], end[1]] = occupant.idx
        self.__board[start[0], start[1]] = 0


class ChessDetection1(Operation):
    name = "Chess Game Test"
    description = "Testing Chess Game"
    InitializeInSeparateThread = True

    def __init__(self):
        super().__init__()
        self.__game = ChessGame()
        self.__conf = NewSlider(
            minimum=1,
            maximum=100,
            step=1,
            name='Confidence',
            default=40,
        )
        self.__show_grid = Boolean(
            name='Draw Grid on Frame',
            label_true='Yes',
            label_false='No'
        )
        self.__show_centers = Boolean(
            name='Draw Centers on Frame',
            label_true='Yes',
            label_false='No'
        )
        self.params.append(self.__conf)
        self.params.append(self.__show_grid)
        self.params.append(self.__show_centers)
        self.__model = YOLO('assets/chess_detection.pt')
        self.__model.cuda(0)
        self.__counter = 0
        self.__grid = {}
        self.__initial = True

    def execute(self, frame: np.ndarray) -> np.ndarray:
        show_grid = self.__show_grid.status
        show_centers = self.__show_centers.status

        self.__counter += 1
        h, w, d = frame.shape
        # noinspection PyTypeChecker
        results: list[Results] = self.__model(
            source=frame,
            device='0',
            imgsz=(h, w),
            verbose=False,
            conf=self.__conf.number / 100
        )

        if self.__counter > 20:
            self.__counter = 0
            local = results[0].to('cpu')
            boxes = self.__finalize_boxes(local)
            if self.__initial:
                try:
                    self.__grid.update(self.__initial_board(boxes))
                    self.__grid['centers'] = self.__compute_centers()
                    self.__initial = False
                except Exception as e:
                    print(e)
            else:
                self.__compute_locations(boxes)

        if show_grid:
            self.__draw_grid(frame)
        if show_centers:
            self.__draw_centers(frame)

        return results[0].plot(conf=False, line_width=1, labels=False)

    @staticmethod
    def __finalize_boxes(results: Results) -> np.ndarray:
        boxes = np.array(results.boxes.xywh)
        cls = np.array(results.boxes.cls)
        conf = np.array(results.boxes.conf)
        boxes_white = boxes[cls == 1, :]
        boxes_black = boxes[cls == 0, :]
        idx_white = np.argsort(conf[cls == 1])[::-1]
        idx_black = np.argsort(conf[cls == 0])[::-1]
        if len(idx_white) > 16:
            new_white = boxes_white[idx_white[:16]]
        else:
            new_white = boxes_white
        if len(idx_black) > 16:
            new_black = boxes_black[idx_black[:16]]
        else:
            new_black = boxes_black
        final = np.zeros((len(new_black) + len(new_white), 5))
        final[0:len(new_white), 0:4] = new_white
        final[len(new_white):, 0:4] = new_black
        final[0:len(new_white), 4] = 1
        final[len(new_white):, 4] = 0
        return final

    def __draw_grid(self, frame: np.ndarray) -> None:
        if 'rows' not in self.__grid or 'cols' not in self.__grid:
            return
        for row in self.__grid['rows']:
            cv2.line(
                frame,
                np.intp((self.__grid['cols'][0, 0], row)),
                np.intp((self.__grid['cols'][1, 7], row)),
                (0, 0, 255),
                1
            )
        for i in range(self.__grid['cols'].shape[1]):
            cv2.line(
                frame,
                np.intp((self.__grid['cols'][0, i], self.__grid['rows'][0])),
                np.intp((self.__grid['cols'][1, i], self.__grid['rows'][-1])),
                (0, 0, 255),
                1
            )

    def __draw_centers(self, frame: np.ndarray) -> None:
        if 'centers' not in self.__grid:
            return
        for row in range(8):
            for column in range(8):
                cv2.circle(
                    frame,
                    np.intp(self.__grid['centers'][row, column, :]),
                    3,
                    (0, 255, 0),
                    -1
                )

    @staticmethod
    def __initial_board(boxes: np.ndarray) -> dict[str, np.ndarray]:
        if boxes.shape[0] != 32:
            raise Exception('Invalid number of boxes!')

        idx = np.argsort(boxes[:, 1])
        r_sorted = boxes[idx, :]
        row_0 = r_sorted[0:8, 1].mean()
        row_1 = r_sorted[8:16, 1].mean()
        row_6 = r_sorted[16:24, 1].mean()
        row_7 = r_sorted[24:32, 1].mean()
        poly = np.polyfit(
            x=[0, 1, 6, 7],
            y=[row_0, row_1, row_6, row_7],
            deg=2
        )
        rows = np.polyval(poly, np.arange(0, 8))

        idx = np.argsort(boxes[:, 0])
        c_sorted = boxes[idx, :]
        csw = c_sorted[c_sorted[:, 4] == 1]
        csb = c_sorted[c_sorted[:, 4] == 0]
        cols = np.zeros((2, 8))
        for i in range(8):
            cols[0, i] = csb[i*2:i*2 + 2, 0].mean()
            cols[1, i] = csw[i*2:i*2 + 2, 0].mean()

        return {'rows': rows, 'cols': cols}

    def __compute_centers(self) -> np.ndarray:
        if 'rows' not in self.__grid or 'cols' not in self.__grid:
            raise Exception('Rows and Cols must be defined')
        grid = np.zeros((8, 8, 2))
        for i in range(8):
            grid[:, i, 0] = np.interp(
                x=self.__grid['rows'],
                xp=(self.__grid['rows'][0], self.__grid['rows'][-1]),
                fp=self.__grid['cols'][:, i],
            )
            grid[:, i, 1] = self.__grid['rows']
        return grid

    def __compute_locations(self, boxes: np.ndarray) -> np.ndarray:
        if 'centers' not in self.__grid:
            raise Exception('Centers are not computed')
        grid = self.__grid['centers'].reshape((8, 8, 2, 1))
        cent = boxes[:, 0:2].reshape((1, 1, 2, boxes.shape[0]))
        dist = np.square(grid - cent).sum(axis=2)
        best = np.min(dist, axis=(0, 1), keepdims=True)
        result = np.equal(dist, best)
        print(result.shape)


class BoardDetection(Operation):
    name = "Board Detection"
    description = "Compute game board grid"

    def __init__(self):
        super().__init__()
        self.__max_corners = IntegerEntry(
            name='Max Corners',
            min_value=1,
            max_value=1000,
            step=5,
            default=25
        )
        self.__test_point = IntegerEntry(
            name='Test Point',
            min_value=1,
            max_value=25,
            step=1,
            default=1
        )
        self.params.append(self.__max_corners)
        self.params.append(self.__test_point)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        idx = self.__test_point.number
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray,
            self.__max_corners.number,
            0.01,
            20,
        )
        vectors = corners - np.swapaxes(corners, 0, 1)
        angles = self.__compute_angles(vectors)

        selected = angles[idx, idx, :, :]
        ideal = np.abs(selected - 90) < 10

        corners = np.intp(corners)
        for i, data in enumerate(corners):
            x, y = data.ravel()
            if i == idx:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(
                frame,
                (x, y),
                2,
                color,
                -1,
            )
        return frame

    @staticmethod
    def __compute_angles(vectors: np.ndarray) -> np.ndarray:
        size = vectors.shape[0]
        mag = np.sqrt(np.sum(np.square(vectors), axis=2))
        mag = mag * mag.T
        dot = np.outer(vectors[:, :, 0], vectors[:, :, 0]) + np.outer(vectors[:, :, 1], vectors[:, :, 1])
        dot = dot.reshape((size, size, size, size))
        result = np.arccos(dot / mag)
        result = np.nan_to_num(result)
        return np.rad2deg(result)


class BoardDetection2(Operation):
    name = 'Board Detection 2'
    description = 'Detect Game Board using Contours'

    def __init__(self):
        super().__init__()
        self.__median = IntegerEntry(
            name='Median Filter K Size',
            min_value=1,
            max_value=25,
            step=2,
            default=5
        )
        self.__block_size = IntegerEntry(
            name='Adaptive Filter Block Size',
            min_value=3,
            max_value=81,
            step=2,
            default=11,
        )
        self.__deviation = NewSlider(
            name='Median Area Filter',
            minimum=0.01,
            maximum=0.25,
            step=0.01,
            default=0.05,
            label_precision=2,
        )
        self.params.append(self.__median)
        self.params.append(self.__block_size)
        self.params.append(self.__deviation)

    def execute(self, frame: np.ndarray) -> np.ndarray:
        median = self.__median.number
        block_size = self.__block_size.number
        deviation = self.__deviation.number
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, median)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            2
        )
        contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        no_noise = [(x, area) for x in contours if (area := cv2.contourArea(x)) > 30]
        accepted_area = []
        accepted_rect = []
        for (cont, area) in no_noise:
            epsilon = 0.025 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            if len(approx) != 4:
                continue
            center, size, angle = cv2.minAreaRect(cont)
            ratio = size[0] / size[1]
            if 0.85 < ratio < 1.15:
                accepted_area.append(area)
                accepted_rect.append(cv2.RotatedRect(center, size, angle))

        median_area = np.median(accepted_area)
        final: list[cv2.RotatedRect] = []
        for (area, rect) in zip(accepted_area, accepted_rect):
            if median_area * 0.65 < area < median_area * 1.5:
                final.append(rect)
        side = np.mean([(x.size[0] + x.size[1]) / 2 for x in final])
        if len(final):
            self.__draw_grid(final[0].center, side, 0, frame)
        return frame

    @staticmethod
    def __draw_grid(center: (float, float), side: float, angle: float, frame: np.ndarray):
        start = (center[0] - side / 2, center[1] - side / 2)
        anchor_x = start[0] - side * ((start[0] + 50) // side)
        anchor_y = start[1] - side * ((start[1] + 50) // side)
        x_start = np.arange(anchor_x, 800, side)
        y_start = np.arange(anchor_y, 800, side)
        cv2.circle(frame, np.intp(center), 3, (0, 255, 0), -1)
        for x in x_start:
            cv2.line(
                frame,
                np.intp([x, anchor_y]),
                np.intp([x, anchor_y + 800]),
                (0, 0, 255),
                1
            )
        for y in y_start:
            cv2.line(
                frame,
                np.intp([anchor_x, y]),
                np.intp([anchor_x + 800, y]),
                (0, 0, 255),
                1
            )
