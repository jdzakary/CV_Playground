from __future__ import annotations

from typing import TYPE_CHECKING
from enum import Enum
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDockWidget, QVBoxLayout, QLabel, QWidget, QApplication, QPushButton
from ultralytics import YOLO

from app.streaming.processing import Operation, NewSlider, Boolean, ButtonGroup

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from app.base import BaseWindow


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
        back = np.array([
            ChessRole.Rook.value, ChessRole.Knight.value, ChessRole.Bishop.value, ChessRole.Queen.value,
            ChessRole.King.value, ChessRole.Bishop.value, ChessRole.Knight.value, ChessRole.Rook.value
        ], dtype=np.uint8)
        self.__board = np.zeros((8, 8, 2), dtype=np.uint8)

        # Setup Black Pieces
        self.__board[0:2, :, 0] = 1
        self.__board[0, :, 1] = back
        self.__board[1, :, 1] = 5

        # Setup White Pieces
        self.__board[6:, :, 0] = 2
        self.__board[7, :, 1] = back
        self.__board[6, :, 1] = 5
        self.__image_board = np.float32(cv2.imread('assets/chess/Board.png'))
        self.__image_white = {
            k.value: cv2.imread(
                f'assets/chess/Piece - White {k.name}.png',
                cv2.IMREAD_UNCHANGED,
            ) for k in ChessRole
        }
        self.__image_black = {
            k.value: cv2.imread(
                f'assets/chess/Piece - Black {k.name}.png',
                cv2.IMREAD_UNCHANGED,
            ) for k in ChessRole
        }

    @property
    def white_pieces(self) -> int:
        return np.sum(self.__board[:, :, 0] == 2)

    @property
    def black_pieces(self) -> int:
        return np.sum(self.__board[:, :, 0] == 1)

    @property
    def board_any(self) -> np.ndarray:
        # noinspection PyTypeChecker
        return self.__board[:, :, 0] > 0

    @property
    def board_white(self):
        return self.__board[:, :, 0] == 2

    @property
    def board_black(self):
        return self.__board[:, :, 0] == 1

    def get_occupant(self, row: int, col: int) -> (int, int):
        item = self.__board[row, col, :]
        if item[0] == 0:
            raise self.EmptySquare(f'No piece at position ({row}, {col})')
        return item

    def is_occupied(self, row: int, col: int) -> bool:
        return self.__board[row, col, 0] != 0

    def __remove_piece(self, row: int, col: int) -> None:
        self.__board[row, col, 0] = 0
        self.__board[row, col, 1] = 0

    def move_piece(self, start: (int, int), end: (int, int)) -> None:
        (player1, role1) = self.get_occupant(*start)
        try:
            (player2, role2) = self.get_occupant(*end)
            if player1 == player2:
                raise Exception('Cannot capture piece of same color')
            self.__remove_piece(*end)
        except self.EmptySquare:
            pass

        self.__remove_piece(*start)
        self.__board[end[0], end[1], 0] = player1
        self.__board[end[0], end[1], 1] = role1

    def create_image(self) -> np.ndarray:
        """
        Taken from https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
        """
        board = self.__image_board
        board = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
        overlay = np.zeros((board.shape[0], board.shape[1], 4))
        idx = np.argwhere(self.__board[:, :, 0])
        for (r, c) in idx:
            player, role = self.__board[r, c, :]
            img = self.__image_white[role] if player == 2 else self.__image_black[role]
            y1 = r * 128
            y2 = (r + 1) * 128
            x1 = c * 128
            x2 = (c + 1) * 128
            overlay[y1:y2, x1:x2] = img
        alpha_channel = overlay[:, :, 3] / 255
        overlay_colors = overlay[:, :, :3]
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
        composite = board * (1 - alpha_mask) + overlay_colors * alpha_mask
        return np.uint8(composite)


class ChessDetection(Operation):
    name = "Chess Game Test"
    description = "Testing Chess Game"
    InitializeInSeparateThread = True

    def __init__(self):
        super().__init__()
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
        self.__buttons = ButtonGroup()
        self.params.append(self.__conf)
        self.params.append(self.__show_grid)
        self.params.append(self.__show_centers)
        self.params.append(self.__buttons)

        self.__game = ChessGame()
        self.__counter = 0
        self.__grid = {}
        self.__initial = True
        self.__memory = np.empty((8, 8, 6), dtype=np.str_)
        self.__rendered_board = np.empty((480, 480), dtype=np.uint8)
        self.__recording = False

        self.__model = YOLO('assets/chess/chess_detection.pt')
        self.__model.cuda(0)

        self.child_functions.append(self.create_dock)
        self.child_functions.append(self.create_buttons)

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

        if self.__counter > 5:
            self.__counter = 0
            local = results[0].to('cpu')
            boxes = self.__finalize_boxes(
                results=local,
                max_white=self.__game.white_pieces,
                max_black=self.__game.black_pieces
            )
            if self.__initial:
                try:
                    self.__grid.update(self.__initial_board(boxes))
                    self.__grid['centers'] = self.__compute_centers()
                    self.__initial = False
                except Exception as e:
                    print(e)
            else:
                try:
                    board = self.__compute_locations(boxes)
                    if self.__check_memory(board):
                        if self.__process_board(board):
                            img = self.__game.create_image()
                            self.__rendered_board = cv2.cvtColor(
                                cv2.resize(img, (480, 480)),
                                cv2.COLOR_RGB2BGR
                            )
                            self.__dock.update_chess(img)
                except Exception as e:
                    print(e)

        if show_grid:
            self.__draw_grid(frame)
        if show_centers:
            self.__draw_centers(frame)

        annotated = results[0].plot(conf=False, line_width=1, labels=False)
        if self.__recording:
            frm = np.empty((480, 640 + 480, 3), dtype=np.uint8)
            frm[:, :640, :] = annotated
            frm[:, 640:, :] = self.__rendered_board
            self.__writer.write(frm)

        return annotated

    @staticmethod
    def __finalize_boxes(
        results: Results,
        max_white: int = 16,
        max_black: int = 16
    ) -> np.ndarray:
        boxes = np.array(results.boxes.xywh)
        cls = np.array(results.boxes.cls)
        conf = np.array(results.boxes.conf)
        boxes_white = boxes[cls == 1, :]
        boxes_black = boxes[cls == 0, :]
        idx_white = np.argsort(conf[cls == 1])[::-1]
        idx_black = np.argsort(conf[cls == 0])[::-1]
        if len(idx_white) > max_white:
            new_white = boxes_white[idx_white[:max_white]]
        else:
            new_white = boxes_white
        if len(idx_black) > max_black:
            new_black = boxes_black[idx_black[:max_black]]
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

        cent = boxes[:, 0:2].transpose()
        cent = cent.reshape((1, 1, 2, boxes.shape[0]))
        dist = np.square(grid - cent).sum(axis=2)
        best = np.min(dist, axis=(0, 1), keepdims=True)
        result = np.equal(dist, best)
        locations = result.sum(axis=2)
        if (locations > 1).sum():
            raise Exception('Multiple pieces on the same square')

        cls = boxes[:, 4].transpose()
        cls = cls.reshape((1, 1, boxes.shape[0]))
        cls = np.repeat(cls, 8, axis=0)
        cls = np.repeat(cls, 8, axis=1)
        cls = cls[result]

        final = np.empty((8, 8), dtype=np.str_)
        final[:, :] = 'O'
        cls_str = np.empty(cls.shape, dtype=np.str_)
        cls_str[cls == 1] = 'W'
        cls_str[cls == 0] = 'B'
        final[locations == 1] = cls_str
        return final

    def __process_board(self, board: np.ndarray) -> bool:
        ideal_white = self.__game.white_pieces
        ideal_black = self.__game.black_pieces
        actual_white = np.sum(board == 'W')
        actual_black = np.sum(board == 'B')

        # No pieces have left the board
        if actual_white == ideal_white and actual_black == ideal_black:
            # Have pieces moved?
            if np.sum(board[self.__game.board_any] == 'O') == 0:
                return False
            start = np.argwhere((board == 'O') & self.__game.board_any)
            end = np.argwhere((board != 'O') & ~self.__game.board_any)
            if len(start) == len(end) == 1:
                self.__game.move_piece(tuple(start[0, :]), tuple(end[0, :]))
                return True
            return False

        # A black piece is captured
        if actual_white == ideal_white and actual_black == ideal_black - 1:
            # Has white moved?
            if np.sum((board == 'W') & ~self.__game.board_white) == 0:
                return False
            start = np.argwhere((board == 'O') & self.__game.board_any)
            end = np.argwhere((board == 'W') & self.__game.board_black)
            self.__game.move_piece(tuple(start[0, :]), tuple(end[0, :]))
            return True

        # A white piece is captured
        if actual_black == ideal_black and actual_white == ideal_white - 1:
            # Has black moved?
            if np.sum((board == 'B') & ~self.__game.board_black) == 0:
                return False
            start = np.argwhere((board == 'O') & self.__game.board_any)
            end = np.argwhere((board == 'B') & self.__game.board_white)
            self.__game.move_piece(tuple(start[0, :]), tuple(end[0, :]))
            return True

        return False

    def __check_memory(self, board: np.ndarray) -> bool:
        self.__memory[:, :, 1:] = self.__memory[:, :, :-1]
        self.__memory[:, :, 0] = board
        return np.all(np.equal(self.__memory, board.reshape((8, 8, 1))))

    def create_dock(self) -> None:
        app: QApplication = QApplication.instance()
        window: BaseWindow = [x for x in app.topLevelWidgets() if type(x).__name__ == "BaseWindow"][0]
        self.__dock = DisplayDock(window)
        window.addDockWidget(Qt.RightDockWidgetArea, self.__dock)
        self.__dock.setFloating(True)
        img = self.__game.create_image()
        self.__rendered_board = cv2.cvtColor(
            cv2.resize(img, (480, 480)),
            cv2.COLOR_RGB2BGR
        )
        self.__dock.update_chess(img)
        self.__dock.toggleRecording.connect(self.__toggle_recording)

    def create_buttons(self) -> None:
        btn1 = self.__buttons.add_button('Show Chess Board')
        btn1.clicked.connect(self.__show_dock)

    def __show_dock(self) -> None:
        self.__dock.setHidden(False)

    def __toggle_recording(self) -> None:
        old = self.__recording
        self.__recording = not self.__recording
        if old:
            self.__writer.release()
        else:
            # TODO: Fix this to use actual frame rate and frame size
            self.__writer = cv2.VideoWriter(
                self.__dock.file_name,
                cv2.VideoWriter.fourcc(*'mp4v'),
                30,
                (640 + 480, 480),
                True,
            )


class DisplayDock(QDockWidget):
    toggleRecording = pyqtSignal(name='toggleRecording')

    def __init__(self, parent):
        super().__init__("View Chess Game", parent)
        self.__recording = False
        self.__file_name = 'chess_game.mp4'

        content = QWidget(self)
        layout = QVBoxLayout()

        self.__image = QLabel()
        self.__save_video = QPushButton("Start Recording")
        self.__save_video.clicked.connect(self.__toggle_recording)

        layout.addWidget(self.__save_video)
        layout.addWidget(self.__image)

        content.setLayout(layout)
        self.setWidget(content)

    @property
    def file_name(self) -> str:
        return self.__file_name

    def update_chess(self, image: np.ndarray) -> None:
        # noinspection DuplicatedCode
        h, w, ch = image.shape
        bytes_per_line = ch * w
        image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap()
        pixmap: QPixmap = pixmap.fromImage(image, Qt.AutoColor)
        ratio = self.parent().devicePixelRatio()
        pixmap = pixmap.scaledToWidth(int(500 * ratio))
        pixmap.setDevicePixelRatio(ratio)
        self.__image.setPixmap(pixmap)

    def __toggle_recording(self) -> None:
        self.__save_video.setText(f'{"Start" if self.__recording else "Stop"} Recording')
        self.__recording = not self.__recording
        self.toggleRecording.emit()
