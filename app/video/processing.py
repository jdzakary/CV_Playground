from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QHBoxLayout

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
        return self.__params


class Parameter(ABC):

    @property
    @abstractmethod
    def component(self) -> QWidget:
        pass


class Slider(Parameter):
    def __init__(self, minimum: int, maximum: int, step: int = 1):
        self.__slider = QSlider()
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(minimum)
        self.slider.setTickInterval(step)
        self.slider.setSingleStep(step)
        self.slider.valueChanged.connect(self.__change_slider)
        self.slider.setOrientation(Qt.Horizontal)

        self.__number = minimum
        self.__component = QWidget(None)
        self.__minimum = Label(f'{minimum}', LabelLevel.P)
        self.__maximum = Label(f'{maximum}', LabelLevel.P)

        l1 = QHBoxLayout()
        l1.addWidget(self.minimum)
        l1.addWidget(self.slider)
        l1.addWidget(self.maximum)
        self.component.setLayout(l1)

    @property
    def component(self) -> QWidget:
        return self.__component

    @property
    def slider(self) -> QSlider:
        return self.__slider

    @property
    def number(self) -> int:
        return self.__number

    @number.setter
    def number(self, value: int):
        self.__number = value

    @property
    def minimum(self) -> Label:
        return self.__minimum

    @minimum.setter
    def minimum(self, value: int):
        if int(self.maximum.text()) <= value:
            raise Exception('Minimum cannot exceed maximum!')
        if self.__slider.value() <= value:
            self.__slider.setValue(value)
        self.__slider.setMinimum(value)
        self.__minimum.setText(f'{value}')

    @property
    def maximum(self):
        return self.__maximum

    @maximum.setter
    def maximum(self, value):
        if int(self.minimum.text()) >= value:
            raise Exception('Maximum cannot be lower than minimum!')
        if self.__slider.value() >= value:
            self.__slider.setValue(value)
        self.__slider.setMaximum(value)
        self.__maximum.setText(f'{value}')

    def __change_slider(self, value: int):
        self.number = value
