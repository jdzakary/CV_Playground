from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSlider, QHBoxLayout, QSpinBox

from app.general.text import Label
from app.general.enums import LabelLevel
from app.video.processing import Operation


class CaptureThread(QThread):

    def __init__(self, parent, process_thread: ProcessThread):
        super().__init__(parent=parent)

        # Property Setup
        self.__video_index = 0
        self.__capture = cv2.VideoCapture(self.video_index)
        self.__process_thread = process_thread

    @property
    def video_index(self) -> int:
        return self.__video_index

    @video_index.setter
    def video_index(self, value: int) -> None:
        if value < 0:
            raise Exception('Invalid Video Index')
        self.__video_index = value
        self.__capture = cv2.VideoCapture(self.__video_index)

    @property
    def capture(self) -> cv2.VideoCapture:
        return self.__capture

    def run(self):
        self.__capture = self.__setup_capture()
        while True:
            ret, frame = self.capture.read()
            if ret:
                self.__process_thread.frame_stack.appendleft(frame)

    @pyqtSlot(int, name='change_device')
    def change_device(self, value: int) -> None:
        self.video_index = value

    def __setup_capture(self) -> cv2.VideoCapture:
        while True:
            capture = cv2.VideoCapture(self.video_index)
            ret, frame = capture.read()
            if ret:
                break
        return capture


class ProcessThread(QThread):
    update_image = pyqtSignal(QImage, name='update_image')
    frame_stack_depth = pyqtSignal(int, name='frame_stack_depth')

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.__frame_stack = deque()
        self.__video_width = 800
        self.__operations: list[Operation] = []

    @property
    def frame_stack(self) -> deque[np.ndarray]:
        return self.__frame_stack

    @property
    def video_width(self) -> int:
        return self.__video_width

    @video_width.setter
    def video_width(self, value: int) -> None:
        if value < 100:
            raise Exception('Width too small!')
        self.__video_width = value

    @property
    def operations(self) -> list[Operation]:
        return self.__operations

    def run(self):
        previous = time.time()
        while True:
            try:
                frame = self.frame_stack.pop()
                for i in self.operations:
                    frame = i.execute(frame)
                picture = self.create_picture(frame)
                self.update_image.emit(picture)
            except IndexError:
                pass
            finally:
                if time.time() - previous > 1:
                    previous = time.time()
                    self.frame_stack_depth.emit(len(self.frame_stack))

    def create_picture(self, frame: np.ndarray) -> QImage:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_format.scaled(self.video_width, self.video_width, Qt.KeepAspectRatio)
        return p

    @pyqtSlot(int, name='change_width')
    def change_width(self, value: int) -> None:
        self.video_width = value

    def change_operations(self, ops: list[Operation]) -> None:
        self.__operations = ops


class StreamControls(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__slider = QSlider(self)
        self.__spinner = QSpinBox(self)
        self.__frame_stack_depth = Label('Frames in Stack: 0', LabelLevel.P)

        # Widget Setup
        self.__setup()

    @property
    def slider(self) -> QSlider:
        return self.__slider

    @property
    def spinner(self) -> QSpinBox:
        return self.__spinner

    @property
    def frame_stack_depth(self) -> Label:
        return self.__frame_stack_depth

    def __control_1(self, gap: int) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.slider.setRange(500, 1500)
        self.slider.setSingleStep(10)
        self.slider.setValue(800)
        self.slider.setOrientation(Qt.Horizontal)
        label = Label('Adjust Video Size', LabelLevel.H4)
        layout.addWidget(label)
        layout.addWidget(self.slider)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(gap)
        return layout

    def __control_2(self, gap: int) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.spinner.setMinimum(0)
        self.spinner.setValue(0)
        label = Label('Select Capture Device:', LabelLevel.H4)
        layout.addWidget(label)
        layout.addWidget(self.spinner)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(gap)
        return layout

    def __stats(self, gap: int) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.frame_stack_depth)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(gap)
        return layout

    def __setup(self, gap: int = 50):
        l1 = QVBoxLayout()
        l1.addLayout(self.__control_1(gap))
        l1.addLayout(self.__control_2(gap))
        l1.addLayout(self.__stats(gap))
        self.setLayout(l1)

    @pyqtSlot(int, name='frame_stack_depth')
    def frame_stack_update(self, value: int):
        self.frame_stack_depth.setText(f'Frames in Stack: {value}')


class StreamWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__video = QLabel(self)
        self.__process = ProcessThread(self)
        self.__capture = CaptureThread(self, self.process)
        self.__controls = StreamControls(self)

        # Widget Setup
        self.__setup()
        self.__start_threads()
        self.show()

    @property
    def video(self) -> QLabel:
        return self.__video

    @property
    def capture(self) -> CaptureThread:
        return self.__capture

    @property
    def process(self) -> ProcessThread:
        return self.__process

    @property
    def controls(self) -> StreamControls:
        return self.__controls

    @pyqtSlot(QImage, name='update_image')
    def update_image(self, image: QImage):
        pixmap = QPixmap()
        self.video.setPixmap(pixmap.fromImage(image, Qt.AutoColor))

    def __start_threads(self) -> None:
        self.controls.slider.valueChanged.connect(self.process.change_width)
        self.controls.spinner.valueChanged.connect(self.capture.change_device)
        self.process.update_image.connect(self.update_image)
        self.process.frame_stack_depth.connect(self.controls.frame_stack_update)
        self.capture.start()
        self.process.start()

    def __setup(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self.controls)
        layout.addWidget(self.video)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.setLayout(layout)
