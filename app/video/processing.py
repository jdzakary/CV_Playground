from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QHBoxLayout, QButtonGroup, QRadioButton, QLabel

from app.config import setting
from app.general.enums import LabelLevel
from app.general.text import Label


class Operation(ABC):
    """
    Abstract Class for any CV/IP operation that is performed
    on the frame stream. All operations must implant the required
    properties, attributes, and methods
    """

    def __init__(self):
        self.__params: list[Parameter] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Defines the name of the Image Processing Operation.
        Should be unique across all operations
        :return:
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        The description of the image processing operation.
        Should be informative but not very long.
        :return:
        """
        pass

    @abstractmethod
    def execute(self, frame: np.ndarray) -> np.ndarray:
        """
        Main logic of the operation. Receives a frame from the processing
        pipeline and must return that frame for the next operation in the pipeline
        :param frame: OpenCV Image Frame (BGR colors)
        :return: Processed image (BGR colors).
        """
        pass

    @staticmethod
    def frame_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image frame to standard RGB format
        :param frame:
        :return:
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def frame_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
        """
        Convert a standard RGB image frame to opencv BGR format
        :param frame:
        :return:
        """
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @property
    def params(self) -> list[Parameter]:
        """
        Optional parameters that are presented to the GUI
        :return:
        """
        return self.__params


class Parameter(ABC):
    """
    Abstract class for parameters of CV Algorithm.
    Must provide a component that is rendered in the tool window.
    Intended to be very flexible
    """

    @property
    @abstractmethod
    def component(self) -> QWidget:
        """
        The Component to be rendered in the tool window
        :return: A pyqt Widget
        """
        pass


class Slider(Parameter):
    """
    A numeric variable that can be adjusted using a slider.
    Qt Sliders can only use integer values. If
    """
    def __init__(self, minimum: int, maximum: int, step: int = 1):
        self.__slider = QSlider()
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(minimum)
        self.slider.setTickInterval(step)
        self.slider.setSingleStep(step)
        self.slider.valueChanged.connect(self.__change_slider)
        self.slider.setOrientation(Qt.Horizontal)

        self.__number = Label(f'{minimum}', LabelLevel.P)
        self.__component = QWidget(None)
        self.__minimum = Label(f'{minimum}', LabelLevel.P)
        self.__maximum = Label(f'{maximum}', LabelLevel.P)

        l1 = QHBoxLayout()
        l1.addWidget(Label('Current Value:', LabelLevel.P))
        l1.addWidget(self.__number)
        l1.addSpacing(35)
        l1.addWidget(self.minimum)
        l1.addWidget(self.slider)
        l1.addWidget(self.maximum)
        self.component.setLayout(l1)

    @property
    def component(self) -> QWidget:
        return self.__component

    @property
    def slider(self) -> QSlider:
        """
        Refernce to the slider object
        :return:
        """
        return self.__slider

    @property
    def number(self) -> int:
        """
        The value of the parameter
        :return:
        """
        return int(self.__number.text())

    @number.setter
    def number(self, value: int):
        self.__number.setText(f'{value}')

    @property
    def minimum(self) -> Label:
        """
        The label object that communicates the minimum
        :return:
        """
        return self.__minimum

    @minimum.setter
    def minimum(self, value: int):
        """
        Changes the minimum value.
        This involves changing the label text, setting the slider min,
        and possibly changing the slider value.
        :param value:
        :return:
        """
        if int(self.maximum.text()) <= value:
            raise Exception('Minimum cannot exceed maximum!')
        if self.__slider.value() <= value:
            self.__slider.setValue(value)
        self.__slider.setMinimum(value)
        self.__minimum.setText(f'{value}')

    @property
    def maximum(self):
        """
        The label object that communicates the maximum
        :return:
        """
        return self.__maximum

    @maximum.setter
    def maximum(self, value):
        """
        Changes the maximum value.
        This involves changing the label text, setting slider max,
        and possible updating the slider value.
        :param value:
        :return:
        """
        if int(self.minimum.text()) >= value:
            raise Exception('Maximum cannot be lower than minimum!')
        if self.__slider.value() >= value:
            self.__slider.setValue(value)
        self.__slider.setMaximum(value)
        self.__maximum.setText(f'{value}')

    def __change_slider(self, value: int):
        """
        Callback connected to the slider.valueChanged slot
        :param value:
        :return:
        """
        self.number = value


class Boolean(Parameter):
    def __init__(self, label_true: str, label_false: str):
        self.__component = QWidget(None)
        self.__button_true = QRadioButton(label_true)
        self.__button_false = QRadioButton(label_false)
        self.__button_true.clicked.connect(self.__change_button)
        self.__button_false.clicked.connect(self.__change_button)
        self.__status = False
        self.status = False

        l1 = QHBoxLayout()
        l1.addWidget(self.__button_true)
        l1.addWidget(self.__button_false)
        self.component.setLayout(l1)

        setting.add_font_callback(self.adjust_fonts)
        self.adjust_fonts()

    @property
    def component(self) -> QWidget:
        return self.__component

    @property
    def status(self) -> bool:
        return self.__status

    @status.setter
    def status(self, value: bool) -> None:
        self.__status = value
        self.__button_false.setChecked(not value)
        self.__button_true.setChecked(value)

    def __change_button(self):
        self.status = not self.status

    def adjust_fonts(self) -> None:
        font = setting.fonts[LabelLevel.P].generate_q()
        try:
            self.__button_false.setFont(font)
            self.__button_true.setFont(font)
        except RuntimeError:
            setting.remove_font_callback(self.adjust_fonts)
