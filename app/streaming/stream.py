from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog, QSizePolicy,
)

from app.config import setting
from app.data.general import signal_manager
from app.data.steaming import StreamDataManager, FrameStack, stream_operations
from app.general.loading_spinner import LoadingSpinner
from app.general.number_spinner import CustomSpinBox
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
            # TODO: Use Direct Show Backend for faster startup times
            # The problem is that Direct Show mysteriously breaks the VideoWriter.
            capture = cv2.VideoCapture(self.__video_index)
            # TODO: Add support to change the camera resolution
            # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
    frameAdded = pyqtSignal(np.ndarray, name='frameAdded')

    def __init__(self, parent):
        super().__init__(parent=parent)

        # Property Setup
        self.__stream_data = StreamDataManager()
        self.__exit = False
        self.__capture: cv2.VideoCapture
        self.__stream_data.replace_listener('toggleStream', self.__toggle_stream)
        self.__stream_data.replace_listener('setVideoIndex', self.__set_video_index)
        self.__frame_stack = FrameStack(
            signal_add=self.frameAdded,
            auto_remove=True
        )

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
            if ret and len(self.__frame_stack) < 3:
                self.__frame_stack.add_frame(frame, True)
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
    update_image = pyqtSignal(QPixmap, name='update_image')
    frame_stack_depth = pyqtSignal(int, name='frame_stack_depth')
    update_latency = pyqtSignal(list, name='update_latency')
    update_fps = pyqtSignal(float, name='update_fps')
    frameRemoved = pyqtSignal(name='frameRemoved')

    def __init__(self, parent: StreamWidget):
        super().__init__(parent=parent)

        # Register Signals
        signal_manager['update_image'] = self.update_image
        signal_manager['frame_stack_depth'] = self.frame_stack_depth
        signal_manager['update_latency'] = self.update_latency
        signal_manager['update_fps'] = self.update_fps

        self.__stream_data = StreamDataManager()
        self.__frame_stack = FrameStack(
            signal_remove=self.frameRemoved,
            auto_add=True
        )
        self.__exit = False
        self.__latency: dict[str, SimpleAvg] = {}
        self.__operations: list[Operation] = []
        self.__writer = self.__create_writer(True)
        self.__stream_data.replace_listener('toggleRecording', self.set_recording)

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
                frame = self.__frame_stack.remove_frame(True)
            except IndexError:
                pass
            else:
                for i in stream_operations:
                    a = time.time_ns()
                    frame = i.execute(frame)
                    b = time.time_ns()
                    try:
                        self.__latency[i.name].update((b - a) / 1e6)
                    except KeyError:
                        self.__latency[i.name] = SimpleAvg(i.name, 20)

                if self.__stream_data.recording:
                    self.__writer.write(frame)

                if (now := time.time()) - previous_1 > 1:
                    fps = frames / (now - previous_1)
                    frames = 0
                    previous_1 = now
                    status = [self.__latency[o.name].current_value() for o in stream_operations]
                    self.update_latency.emit(status)
                    self.update_fps.emit(fps)

                picture = self.__create_picture(frame)
                self.update_image.emit(picture)
                frames += 1
            finally:
                if (now := time.time()) - previous_2 > 1:
                    previous_2 = now
                    self.frame_stack_depth.emit(len(self.__frame_stack))

    def __create_picture(self, frame: np.ndarray) -> QPixmap:
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

        image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap()
        pixmap: QPixmap = pixmap.fromImage(image, Qt.AutoColor)

        ratio = self.parent().devicePixelRatio()
        pixmap = pixmap.scaledToWidth(int(self.__stream_data.video_width * ratio))
        pixmap.setDevicePixelRatio(ratio)
        return pixmap

    def __create_writer(self, initial: bool = False):
        # TODO: Figure out how to save video when Capture Device uses direct show
        writer = cv2.VideoWriter(
            self.__stream_data.file_name,
            cv2.VideoWriter.fourcc(*'mp4v'),
            self.__stream_data.fps,
            self.__stream_data.frame_size,
            True,
        )
        if initial:
            writer.release()
            try:
                os.remove('video_out.mp4')
            except FileNotFoundError:
                pass
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
        self.__set_video_display_width = CustomSpinBox(self)
        self.__set_video_index = CustomSpinBox(self)
        self.__toggle_recording = QPushButton('Start Recording')
        self.__toggle_streaming = QPushButton('Start Streaming')
        self.__file_location = QPushButton('Change Save Location')
        self.__save_default_width = QPushButton('Save As Default')
        self.__frame_stack_depth = Label('Frames in Stack: 0', LabelLevel.P)
        self.__fps = Label('FPS: 0', LabelLevel.P)
        self.__streaming_loading = LoadingSpinner(self)

        # Replace Listeners
        self.__stream_data.replace_listener('toggleRecording', self.__set_recording)
        self.__stream_data.replace_listener('toggleStream', self.__set_stream)
        self.__stream_data.replace_listener('setDeviceLoading', self.__set_device_loading)

        # Proxy Signals
        self.__toggle_streaming.clicked.connect(self.__proxy_toggle_stream)
        self.__toggle_recording.clicked.connect(self.__proxy_toggle_recording)
        self.__save_default_width.clicked.connect(self.__proxy_save_width)

        # Register Signals
        signal_manager['setVideoWidth'] = self.__set_video_display_width.newValue
        signal_manager['setVideoIndex'] = self.__set_video_index.newValue
        signal_manager['launchFileDialog'] = self.__file_location.clicked
        signal_manager['toggleStream'] = self.toggleStream
        signal_manager['toggleRecording'] = self.toggleRecording

        # Connect Slots to Signals
        signal_manager['frame_stack_depth'].connect(self.frame_stack_update)
        signal_manager['update_fps'].connect(self.__update_fps)

        # Widget Setup
        self.__setup(gap=20)

    def __control_1(self, gap: int) -> QHBoxLayout:
        """
        Control set 1:
            Layout for the slider that adjusts streaming size
        :param gap:
        :return:
        """
        layout = QHBoxLayout()
        self.__set_video_display_width.setRange(200, 800)
        self.__set_video_display_width.setSingleStep(4)
        self.__set_video_display_width.setValue(self.__stream_data.video_width)

        layout.addWidget(Label('Display Size', LabelLevel.H4))
        layout.addWidget(self.__set_video_display_width)
        layout.addWidget(self.__save_default_width)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(gap)
        return layout

    def __control_2(self, gap: int) -> QHBoxLayout:
        """
        Control set 2:
            LoadingSpinner that selects the capture device.
            Button to toggle recording
            File picker to choose recording output
        :param gap:
        :return:
        """
        layout = QHBoxLayout()
        self.__set_video_index.setMinimum(0)
        self.__set_video_index.setValue(0)
        self.__toggle_recording.setDisabled(True)
        self.__streaming_loading.hide()
        l1 = QHBoxLayout()
        l1.addWidget(self.__toggle_streaming)
        l1.addWidget(self.__streaming_loading)
        l1.setSpacing(5)

        layout.addWidget(Label('Select Capture Device:', LabelLevel.H4))
        layout.addWidget(self.__set_video_index)
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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setLayout(l1)

    def frame_stack_update(self, value: int):
        self.__frame_stack_depth.setText(f'Frames in Stack: {value}')

    def __set_recording(self, value: bool) -> None:
        if value:
            self.__toggle_recording.setText('Stop Recording')
            self.__set_video_index.setDisabled(True)
            self.__file_location.setDisabled(True)
            self.__toggle_streaming.setDisabled(True)
        else:
            self.__toggle_recording.setText('Start Recording')
            self.__set_video_index.setDisabled(False)
            self.__file_location.setDisabled(False)
            self.__toggle_streaming.setDisabled(False)
        self.__stream_data.set_recording(value)

    def __set_stream(self, value: bool) -> None:
        self.__toggle_streaming.setText('Stop Streaming' if value else 'Start Streaming')
        self.__toggle_recording.setEnabled(value)
        self.__stream_data.set_streaming(value)

    def __proxy_toggle_stream(self) -> None:
        self.toggleStream.emit(not self.__stream_data.streaming)

    def __proxy_toggle_recording(self) -> None:
        if not self.__stream_data.device_loading:
            self.toggleRecording.emit(not self.__stream_data.recording)

    def __proxy_save_width(self) -> None:
        setting.streaming['video_display_width'] = self.__set_video_display_width.value()

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
        self.__capture = CaptureThread(self)
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

    @pyqtSlot(QPixmap, name='update_image')
    def update_image(self, pixmap: QPixmap):
        """
        Callback to update the pixmap of the streaming object
        :param image:
        :return:
        """
        self.__video.setPixmap(pixmap)

    def __start_threads(self) -> None:
        """
        Thread related logic
        :return:
        """
        self.__capture.start(QThread.HighPriority)
        self.__process.start(QThread.HighestPriority)

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
            filter='Videos (*.mp4)',
        )
        if file_name:
            self.setFileName.emit(file_name)

    def __set_video_placeholder(self) -> None:
        pixmap = QPixmap('assets/camera_loading.png')
        pixmap = pixmap.scaledToWidth(400)
        pixmap.setDevicePixelRatio(self.devicePixelRatio())
        self.__video.setPixmap(pixmap)
