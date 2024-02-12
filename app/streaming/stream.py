from __future__ import annotations

import os
import time
from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSlider,
    QHBoxLayout, QSpinBox, QPushButton, QFileDialog,
)

from app.data.general import signal_manager
from app.data.steaming import StreamDataManager
from app.general.spinner import Spinner
from app.general.stats import SimpleAvg
from app.general.text import Label
from app.general.enums import LabelLevel
from app.streaming.processing import Operation


class CreateCaptureDevice(QThread):
    captureDeviceCreated = pyqtSignal(cv2.VideoCapture, name='captureDeviceCreated')

    def __init__(self, video_index: int, parent) -> None:
        super().__init__(parent=parent)
        self.__video_index = video_index

    def run(self) -> None:
        while True:
            capture = cv2.VideoCapture(self.__video_index)
            ret, frame = capture.read()
            if ret:
                break
        self.captureDeviceCreated.emit(capture)


class CaptureThread(QThread):
    """
    Thread responsible for reading raw frames from a camera.
    """

    setFPS = pyqtSignal(float, name='setFPS')
    setFrameSize = pyqtSignal(tuple, name='setFrameSize')
    setDeviceLoading = pyqtSignal(bool, name='setDeviceLoading')

    def __init__(self, parent, process_thread: ProcessThread):
        super().__init__(parent=parent)

        # Property Setup
        self.__stream_data = StreamDataManager()
        self.__exit = False
        self.__capture: cv2.VideoCapture
        self.__process_thread = process_thread
        self.__stream_data.replace_listener('toggleStream', self.__toggle_stream)
        self.__stream_data.replace_listener('setVideoIndex', self.__set_video_index)

        # Register Signals
        signal_manager['setFPS'] = self.setFPS
        signal_manager['setFrameSize'] = self.setFrameSize
        signal_manager['setDeviceLoading'] = self.setDeviceLoading

    def run(self):
        """
        Executed by Qt Thread manager.
        Set up the capture device, then continue to read frames for infinity
        :return:
        """
        while True:
            if self.__exit:
                break
            if not self.__stream_data.streaming:
                continue
            ret, frame = self.__capture.read()
            if ret and len(self.__process_thread.frame_stack) < 3:
                self.__process_thread.frame_stack.appendleft(frame)
            if not ret:
                signal_manager['toggleRecording'].emit(False)
                signal_manager['toggleStream'].emit(False)

    # @pyqtSlot(int, name='changeVideoDevice')
    def __set_video_index(self, value: int) -> None:
        """
        Qt Slot to change the capture device
        :param value:
        :return:
        """
        if self.__stream_data.recording:
            raise Exception('Cannot Change device while recording!')
        if value < 0:
            raise Exception('Invalid Video Index')
        self.__stream_data.set_video_index(value)
        if self.__stream_data.streaming:
            signal_manager['toggleStream'].emit(False)
            signal_manager['toggleStream'].emit(True)

    # @pyqtSlot(bool, name='toggleStream')
    def __toggle_stream(self, value: bool) -> None:
        if value:
            self.setDeviceLoading.emit(True)
            thread = CreateCaptureDevice(self.__stream_data.video_index, self)
            thread.captureDeviceCreated.connect(self.__receive_capture)
            thread.start()
        else:
            try:
                self.__stream_data.set_streaming(False)
                self.__capture.release()
            except AttributeError:
                pass

    def exit(self, returnCode=0):
        self.__exit = True
        try:
            self.__capture.release()
        except AttributeError:
            pass
        super().exit(returnCode)

    def __receive_capture(self, capture: cv2.VideoCapture) -> None:
        rows = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cols = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.setFPS.emit(fps)
        self.setFrameSize.emit((cols, rows))
        self.__capture = capture
        self.__stream_data.set_streaming(True)
        self.setDeviceLoading.emit(False)


class ProcessThread(QThread):
    """
    Thread responsible for applying the real-time CV operations.
    After all operations, sends image data to be displayed by the main thread.
    """
    update_image = pyqtSignal(QImage, name='update_image')
    frame_stack_depth = pyqtSignal(int, name='frame_stack_depth')
    update_latency = pyqtSignal(list, name='update_latency')
    update_fps = pyqtSignal(float, name='update_fps')

    def __init__(self, parent: StreamWidget):
        super().__init__(parent=parent)

        # Register Signals
        signal_manager['update_image'] = self.update_image
        signal_manager['frame_stack_depth'] = self.frame_stack_depth
        signal_manager['update_latency'] = self.update_latency
        signal_manager['update_fps'] = self.update_fps

        self.__stream_data = StreamDataManager()
        self.__frame_stack = deque()
        self.__exit = False
        self.__latency: dict[str, SimpleAvg] = {}
        self.__operations: list[Operation] = []
        self.__writer = self.__create_writer(True)
        self.__stream_data.replace_listener('toggleRecording', self.set_recording)

    @property
    def frame_stack(self) -> deque[np.ndarray]:
        """
        The frames waiting to be processed
        :return:
        """
        return self.__frame_stack

    @property
    def operations(self) -> list[Operation]:
        """
        The list of operations to be applied
        :return:
        """
        return self.__operations

    def run(self):
        """
        Main execute method called by Qt Thread manager.
        Apply all operations and then update picture.
        :return:
        """
        previous_1 = time.time()
        previous_2 = time.time()
        frames = 0
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

                if self.__stream_data.recording:
                    self.__writer.write(frame)

                if (now := time.time()) - previous_1 > 1:
                    fps = frames / (now - previous_1)
                    frames = 0
                    previous_1 = now
                    status = [self.__latency[o.name].current_value() for o in self.operations]
                    self.update_latency.emit(status)
                    self.update_fps.emit(fps)

                picture = self.__create_picture(frame)
                self.update_image.emit(picture)
                frames += 1
            finally:
                if (now := time.time()) - previous_2 > 1:
                    previous_2 = now
                    self.frame_stack_depth.emit(len(self.frame_stack))

    def __create_picture(self, frame: np.ndarray) -> QImage:
        """
        Convert OpenCV frame to Qt Image Object.
        Qt Image objects are applied to a QLabel to create the appearance of a streaming

        Taken From: https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv
        Credit to: https://stackoverflow.com/users/2172752/martencatcher

        :param frame:
        :return:
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_format.scaled(
            self.__stream_data.video_width,
            self.__stream_data.video_width,
            Qt.KeepAspectRatio
        )
        return p

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

    def __create_writer(self, initial: bool = False):
        writer = cv2.VideoWriter(
            self.__stream_data.file_name,
            cv2.VideoWriter.fourcc(*'XVID'),
            self.__stream_data.fps,
            self.__stream_data.frame_size,
        )
        if initial:
            writer.release()
            os.remove('video_out.avi')
            return writer
        else:
            self.__writer = writer

    def set_recording(self, value: bool) -> None:
        if value:
            self.__create_writer()
            self.__stream_data.set_recording(True)
        else:
            self.__stream_data.set_recording(False)
            self.__writer.release()

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

    toggleStream = pyqtSignal(bool, name='toggleStream')
    toggleRecording = pyqtSignal(bool, name='toggleRecording')

    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__stream_data = StreamDataManager()
        self.__slider = QSlider(self)
        self.__spinner = QSpinBox(self)
        self.__toggle_recording = QPushButton('Start Recording')
        self.__toggle_streaming = QPushButton('Start Streaming')
        self.__file_location = QPushButton('Change Save Location')
        self.__frame_stack_depth = Label('Frames in Stack: 0', LabelLevel.P)
        self.__fps = Label('FPS: 0', LabelLevel.P)
        self.__streaming_loading = Spinner(self)

        # Replace Listeners
        self.__stream_data.replace_listener('toggleRecording', self.__set_recording)
        self.__stream_data.replace_listener('toggleStream', self.__set_stream)
        self.__stream_data.replace_listener('setDeviceLoading', self.__set_device_loading)

        # Proxy Signals
        self.__toggle_streaming.clicked.connect(self.__proxy_toggle_stream)
        self.__toggle_recording.clicked.connect(self.__proxy_toggle_recording)

        # Register Signals
        signal_manager['setVideoWidth'] = self.__slider.valueChanged
        signal_manager['setVideoIndex'] = self.__spinner.valueChanged
        signal_manager['launchFileDialog'] = self.__file_location.clicked
        signal_manager['toggleStream'] = self.toggleStream
        signal_manager['toggleRecording'] = self.toggleRecording

        # Connect Slots to Signals
        signal_manager['frame_stack_depth'].connect(self.frame_stack_update)
        signal_manager['update_fps'].connect(self.__update_fps)

        # Widget Setup
        self.__setup()

    def __control_1(self, gap: int) -> QHBoxLayout:
        """
        Control set 1:
            Layout for the slider that adjusts streaming size
        :param gap:
        :return:
        """
        layout = QHBoxLayout()
        self.__slider.setRange(500, 1500)
        self.__slider.setSingleStep(10)
        self.__slider.setValue(800)
        self.__slider.setOrientation(Qt.Horizontal)
        label = Label('Adjust Video Size', LabelLevel.H4)
        layout.addWidget(label)
        layout.addWidget(self.__slider)
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
        self.__spinner.setMinimum(0)
        self.__spinner.setValue(0)
        self.__toggle_recording.setDisabled(True)
        self.__streaming_loading.hide()
        l1 = QHBoxLayout()
        l1.addWidget(self.__toggle_streaming)
        l1.addWidget(self.__streaming_loading)
        l1.setSpacing(5)

        layout.addWidget(Label('Select Capture Device:', LabelLevel.H4))
        layout.addWidget(self.__spinner)
        layout.addLayout(l1)
        layout.addWidget(self.__toggle_recording)
        layout.addWidget(self.__file_location)

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
        layout.addWidget(self.__frame_stack_depth)
        layout.addWidget(self.__fps)
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

    # @pyqtSlot(int, name='frame_stack_depth')
    def frame_stack_update(self, value: int):
        self.__frame_stack_depth.setText(f'Frames in Stack: {value}')

    # @pyqtSlot(bool, name='toggleRecording')
    def __set_recording(self, value: bool) -> None:
        if value:
            self.__toggle_recording.setText('Stop Recording')
            self.__spinner.setDisabled(True)
            self.__file_location.setDisabled(True)
            self.__toggle_streaming.setDisabled(True)
        else:
            self.__toggle_recording.setText('Start Recording')
            self.__spinner.setDisabled(False)
            self.__file_location.setDisabled(False)
            self.__toggle_streaming.setDisabled(False)
        self.__stream_data.set_recording(value)

    # @pyqtSlot(bool, name='toggleStream')
    def __set_stream(self, value: bool) -> None:
        self.__toggle_streaming.setText('Stop Streaming' if value else 'Start Streaming')
        self.__toggle_recording.setEnabled(value)
        self.__stream_data.set_streaming(value)

    def __proxy_toggle_stream(self) -> None:
        self.toggleStream.emit(not self.__stream_data.streaming)

    def __proxy_toggle_recording(self) -> None:
        self.toggleRecording.emit(not self.__stream_data.recording)

    def __set_device_loading(self, value: bool) -> None:
        if value:
            self.__toggle_streaming.setDisabled(True)
            self.__streaming_loading.setHidden(False)
        else:
            self.__toggle_streaming.setDisabled(False)
            self.__streaming_loading.setHidden(True)
        self.__stream_data.set_device_loading(value)

    def __update_fps(self, value: float) -> None:
        self.__fps.setText(f'FPS: {value:.3f}')


class StreamWidget(QWidget):
    """
    Primary widget for the stream tab
    """
    setFileName = pyqtSignal(str, name='setFileName')

    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__stream_data = StreamDataManager()
        self.__video = QLabel(self)
        self.__process = ProcessThread(self)
        self.__capture = CaptureThread(self, self.__process)
        self.__controls = StreamControls(self)

        # Register Signals
        signal_manager['setFileName'] = self.setFileName

        # Connect Slots to Signals
        signal_manager['update_image'].connect(self.update_image)
        signal_manager['launchFileDialog'].connect(self.__change_file)

        # Widget Setup
        self.__setup()
        self.__start_threads()
        self.show()

    @property
    def video(self) -> QLabel:
        """
        The streaming object that displays rapidly changing QImages
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

    # @pyqtSlot(QImage, name='update_image')
    def update_image(self, image: QImage):
        """
        Callback to update the pixmap of the streaming object
        :param image:
        :return:
        """
        pixmap = QPixmap()
        self.__video.setPixmap(pixmap.fromImage(image, Qt.AutoColor))

    def __start_threads(self) -> None:
        """
        Thread related logic
        :return:
        """
        self.__capture.start()
        self.__process.start()

    def __setup(self) -> None:
        """
        General Widget setup
        :return:
        """
        layout = QVBoxLayout()
        layout.addWidget(self.__controls)
        layout.addWidget(self.__video)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.__set_video_placeholder()
        self.setLayout(layout)

    def __change_file(self) -> None:
        # noinspection PyArgumentList
        file_name, _ = QFileDialog.getSaveFileName(
            parent=self,
            caption='Set Output File Name',
            filter='Videos (*.avi)',
        )
        if file_name:
            self.setFileName.emit(file_name)

    def __set_video_placeholder(self) -> None:
        file = QPixmap('assets/camera_loading.png')
        pixmap = file.scaled(
            self.__stream_data.video_width,
            self.__stream_data.video_width,
            Qt.KeepAspectRatio
        )
        self.__video.setPixmap(pixmap)
