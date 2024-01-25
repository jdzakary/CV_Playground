from __future__ import annotations

import copy
import os
import time
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSlider, QHBoxLayout, QSpinBox, QPushButton, QFileDialog

from app.general.stats import SimpleAvg
from app.general.text import Label
from app.general.enums import LabelLevel
from app.video.processing import Operation


class CaptureThread(QThread):
    """
    Thread responsible for reading raw frames from a camera.
    """

    def __init__(self, parent, process_thread: ProcessThread):
        super().__init__(parent=parent)

        # Property Setup
        self.__video_index = 0
        self.__exit = False
        self.__capture = cv2.VideoCapture(self.video_index)
        self.__process_thread = process_thread

    @property
    def video_index(self) -> int:
        """
        OpenCV index of the video captrue device
        :return:
        """
        return self.__video_index

    @video_index.setter
    def video_index(self, value: int) -> None:
        """
        Change the video capture device by setting a new video_index
        :param value:
        :return:
        """
        if value < 0:
            raise Exception('Invalid Video Index')
        self.__video_index = value
        self.__capture = cv2.VideoCapture(self.__video_index)

    @property
    def capture(self) -> cv2.VideoCapture:
        """
        Reference to the capture device
        :return:
        """
        return self.__capture

    def run(self):
        """
        Executed by Qt Thread manager.
        Set up the capture device, then continue to read frames for infinity
        :return:
        """
        self.__capture = self.__setup_capture()
        while True:
            if self.__exit:
                break
            ret, frame = self.capture.read()
            if ret:
                self.__process_thread.frame_stack.appendleft(frame)

    @pyqtSlot(int, name='change_device')
    def change_device(self, value: int) -> None:
        """
        Qt Slot to change the capture device
        :param value:
        :return:
        """
        if self.__process_thread.writing:
            raise Exception('Cannot Change device while recording!')
        self.video_index = value

    def exit(self, returnCode=0):
        self.__exit = True
        self.capture.release()
        super().exit(returnCode)

    def __setup_capture(self) -> cv2.VideoCapture:
        """
        Initial setup for the capture device.
        This prevents program crash if no camera is detected upon initial launch.
        :return:
        """
        while True:
            capture = cv2.VideoCapture(self.video_index)
            ret, frame = capture.read()
            if ret:
                break
        return capture


class ProcessThread(QThread):
    """
    Thread responsible for applying the real-time CV operations.
    After all operations, sends image data to be displayed by the main thread.
    """
    update_image = pyqtSignal(QImage, name='update_image')
    frame_stack_depth = pyqtSignal(int, name='frame_stack_depth')
    update_latency = pyqtSignal(list, name='update_latency')

    def __init__(self, parent: StreamWidget):
        super().__init__(parent=parent)
        self.__frame_stack = deque()
        self.__exit = False
        self.__file_name = 'video_out.avi'
        self.__latency: dict[str, SimpleAvg] = {}
        self.__writing = False
        self.__video_width = 800
        self.__operations: list[Operation] = []
        self.__writer = self.__create_writer(True)

    @property
    def frame_stack(self) -> deque[np.ndarray]:
        """
        The frames waiting to be processed
        :return:
        """
        return self.__frame_stack

    @property
    def video_width(self) -> int:
        """
        Size of the video output to screen
        :return:
        """
        return self.__video_width

    @video_width.setter
    def video_width(self, value: int) -> None:
        """
        Adjust the video output size
        :param value:
        :return:
        """
        if value < 100:
            raise Exception('Width too small!')
        self.__video_width = value

    @property
    def operations(self) -> list[Operation]:
        """
        The list of operations to be applied
        :return:
        """
        return self.__operations

    @property
    def file_name(self) -> str:
        return self.__file_name

    @file_name.setter
    def file_name(self, value: str):
        self.__file_name = value

    @property
    def writing(self) -> bool:
        return self.__writing

    @writing.setter
    def writing(self, value: bool) -> None:
        if value == self.__writing:
            return
        if self.__writing:
            self.__writing = False
            self.__writer.release()
        else:
            self.__create_writer()
            self.__writing = True

    def run(self):
        """
        Main execute method called by Qt Thread manager.
        Apply all operations and then update picture.
        :return:
        """
        previous_1 = time.time()
        previous_2 = time.time()
        while True:
            if self.__exit:
                break
            try:
                frame = self.frame_stack.pop()
            except IndexError:
                pass
            else:
                for i in self.operations:
                    a = time.time_ns()
                    frame = i.execute(frame)
                    b = time.time_ns()
                    self.__latency[i.name].update((b - a) / 1e6)
                if self.writing:
                    self.__writer.write(frame)

                if (now := time.time()) - previous_1 > 1:
                    previous_1 = now
                    status = [self.__latency[o.name].current_value() for o in self.operations]
                    self.update_latency.emit(status)

                picture = self.create_picture(frame)
                self.update_image.emit(picture)
            finally:
                if (now := time.time()) - previous_2 > 1:
                    previous_2 = now
                    self.frame_stack_depth.emit(len(self.frame_stack))

    def create_picture(self, frame: np.ndarray) -> QImage:
        """
        Convert OpenCV frame to Qt Image Object.
        Qt Image objects are applied to a QLabel to create the appearance of a video

        Taken From: https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv
        Credit to: https://stackoverflow.com/users/2172752/martencatcher

        :param frame:
        :return:
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_format.scaled(self.video_width, self.video_width, Qt.KeepAspectRatio)
        return p

    @pyqtSlot(int, name='change_width')
    def change_width(self, value: int) -> None:
        """
        Qt slot to change the width of the video output stream
        :param value:
        :return:
        """
        self.video_width = value

    def change_operations(self, ops: list[Operation]) -> None:
        """
        Callback to replace the list of operations being used.
        Connected to the ManageOperations dock window
        :param ops:
        :return:
        """
        for i in ops:
            if i.name not in self.__latency:
                self.__latency[i.name] = SimpleAvg(i.name, 20)
        self.__operations = ops


    def toggle_writing(self) -> None:
        self.writing = not self.writing

    def __create_writer(self, initial: bool = False):
        four_cc = cv2.VideoWriter.fourcc(*'XVID')

        if initial:
            writer = cv2.VideoWriter(self.file_name, four_cc, 20, (10, 10))
            writer.release()
            os.remove('video_out.avi')
            return writer
        else:
            cap: cv2.VideoCapture = self.parent().capture.capture
            cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = float(cap.get(cv2.CAP_PROP_FPS))
            self.__writer.open(self.file_name, four_cc, frame_rate, (cols, rows))

    def exit(self, returnCode=0):
        self.__exit = True
        try:
            self.__writer.release()
        except AttributeError:
            pass
        super().exit(returnCode)


class StreamControls(QWidget):
    """
    Controls for adjusting the stream and saving to a file.
    """
    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__slider = QSlider(self)
        self.__spinner = QSpinBox(self)
        self.__toggle = QPushButton('Start Recording')
        self.__file_location = QPushButton('Change Save Location')
        self.__frame_stack_depth = Label('Frames in Stack: 0', LabelLevel.P)

        # Widget Setup
        self.toggle.clicked.connect(self.__toggle_text)
        self.__setup()

    @property
    def slider(self) -> QSlider:
        """
        Slider to adjust the video stream output size (No effect on saved file).
        :return:
        """
        return self.__slider

    @property
    def spinner(self) -> QSpinBox:
        """
        Widget to change the OpenCV camera index
        :return:
        """
        return self.__spinner

    @property
    def toggle(self) -> QPushButton:
        return self.__toggle

    @property
    def file_location(self) -> QPushButton:
        return self.__file_location

    @property
    def frame_stack_depth(self) -> Label:
        """
        Represents the number of frames waiting to be processed
        :return:
        """
        return self.__frame_stack_depth

    def __control_1(self, gap: int) -> QHBoxLayout:
        """
        Control set 1:
            Layout for the slider that adjusts video size
        :param gap:
        :return:
        """
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
        """
        Control set 2:
            Spinner that selects the capture device.
            Button to toggle recording
            File picker to choose recording output
        :param gap:
        :return:
        """
        layout = QHBoxLayout()
        self.spinner.setMinimum(0)
        self.spinner.setValue(0)

        layout.addWidget(Label('Select Capture Device:', LabelLevel.H4))
        layout.addWidget(self.spinner)
        layout.addWidget(self.toggle)
        layout.addWidget(self.file_location)

        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(gap)
        return layout

    def __stats(self, gap: int) -> QHBoxLayout:
        """
        Layout for the real-time stats
        :param gap:
        :return:
        """
        layout = QHBoxLayout()
        layout.addWidget(self.frame_stack_depth)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(gap)
        return layout

    def __setup(self, gap: int = 50):
        """
        Overall setup of the widget
        :param gap:
        :return:
        """
        l1 = QVBoxLayout()
        l1.addLayout(self.__control_1(gap))
        l1.addLayout(self.__control_2(gap))
        l1.addLayout(self.__stats(gap))
        self.setLayout(l1)

    @pyqtSlot(int, name='frame_stack_depth')
    def frame_stack_update(self, value: int):
        self.frame_stack_depth.setText(f'Frames in Stack: {value}')

    def __toggle_text(self) -> None:
        if self.toggle.text() == 'Start Recording':
            self.toggle.setText('Stop Recording')
            self.spinner.setDisabled(True)
            self.file_location.setDisabled(True)
        else:
            self.toggle.setText('Start Recording')
            self.spinner.setDisabled(False)
            self.file_location.setDisabled(False)


class StreamWidget(QWidget):
    """
    Primary widget for the stream tab
    """
    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__video = QLabel(self)
        self.__process = ProcessThread(self)
        self.__capture = CaptureThread(self, self.process)
        self.__controls = StreamControls(self)
        self.controls.file_location.clicked.connect(self.__change_file)

        # Widget Setup
        self.__setup()
        self.__start_threads()
        self.show()

    @property
    def video(self) -> QLabel:
        """
        The video object that displays rapidly changing QImages
        :return:
        """
        return self.__video

    @property
    def capture(self) -> CaptureThread:
        """
        Reference to the capture thread
        :return:
        """
        return self.__capture

    @property
    def process(self) -> ProcessThread:
        """
        Reference to the processing thread
        :return:
        """
        return self.__process

    @property
    def controls(self) -> StreamControls:
        """
        Reference to the stream controls
        :return:
        """
        return self.__controls

    @pyqtSlot(QImage, name='update_image')
    def update_image(self, image: QImage):
        """
        Callback to update the pixmap of the video object
        :param image:
        :return:
        """
        pixmap = QPixmap()
        self.video.setPixmap(pixmap.fromImage(image, Qt.AutoColor))

    def __start_threads(self) -> None:
        """
        Thread related logic
        :return:
        """
        self.controls.slider.valueChanged.connect(self.process.change_width)
        self.controls.spinner.valueChanged.connect(self.capture.change_device)
        self.controls.toggle.clicked.connect(self.process.toggle_writing)
        self.process.update_image.connect(self.update_image)
        self.process.frame_stack_depth.connect(self.controls.frame_stack_update)
        self.capture.start()
        self.process.start()

    def __setup(self) -> None:
        """
        General Widget setup
        :return:
        """
        layout = QVBoxLayout()
        layout.addWidget(self.controls)
        layout.addWidget(self.video)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.setLayout(layout)

    def __change_file(self) -> None:
        # noinspection PyArgumentList
        file_name, _ = QFileDialog.getSaveFileName(
            parent=self,
            caption='Set Output File Name',
            filter='Videos (*.avi)',
        )
        if file_name:
            self.process.file_name = file_name
