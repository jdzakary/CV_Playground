from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QHBoxLayout, QButtonGroup, QRadioButton, QVBoxLayout

from app.config import setting
from app.general.enums import LabelLevel
from app.general.layouts import FlowLayout
from app.general.text import Label


class Operation:
    """
    Abstract Class for any CV/IP operation that is performed
    on the frame stream. All operations must implant the required
    properties, attributes, and methods
    """
    name = 'Abstract Operation'
    description = 'Replace me please'
    InitializeInSeparateThread = False

    def __init__(self):
        self.__params: list[Parameter] = []

    def execute(self, frame: np.ndarray) -> np.ndarray:
        """
        Main logic of the operation. Receives a frame from the processing
        pipeline and must return that frame for the next operation in the pipeline
        :param frame: OpenCV Image Frame (BGR colors)
        :return: Processed image (BGR colors).
        """
        pass

    def create_controls(self):
        for i in self.__params:
            i.create_master()

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
    def __init__(self, name: str) -> None:
        self.__name = name

    def create_master(self) -> None:
        self.__component = QWidget(None)
        layout = QVBoxLayout()
        layout.addWidget(Label(self.__name, LabelLevel.H4))
        layout.addWidget(self.create_child())
        self.__component.setLayout(layout)

    @property
    def component(self) -> QWidget:
        """
        The Component to be rendered in the tool window
        :return: A pyqt Widget
        """
        return self.__component

    @abstractmethod
    def create_child(self) -> QWidget:
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
    def __init__(
        self,
        minimum: int,
        maximum: int,
        step: int = 1,
        name: str = '',
        default: int = None,
        divisor: int = 1,
    ) -> None:
        self.__min = minimum
        self.__max = maximum
        self.__step = step
        self.__name = name
        self.__default = default
        self.__divisor = divisor

        super().__init__(name=name)

    def create_child(self) -> QWidget:
        self.__slider = QSlider()
        self.__slider.setRange(self.__min, self.__max)
        self.__slider.setValue(self.__min)
        self.__slider.setTickInterval(self.__step)
        self.__slider.setSingleStep(self.__step)
        self.__slider.valueChanged.connect(self.__change_slider)
        self.__slider.setOrientation(Qt.Horizontal)

        component = QWidget(None)
        self.__number = Label(
            f'{self.__min / self.__divisor}',
            LabelLevel.P
        )
        self.__minimum = Label(
            f'{self.__min / self.__divisor}',
            LabelLevel.P
        )
        self.__maximum = Label(
            f'{self.__max / self.__divisor}',
            LabelLevel.P
        )
        if self.__default is not None:
            self.__slider.setValue(self.__default)

        l1 = QHBoxLayout()
        l1.addWidget(Label('Current Value:', LabelLevel.P))
        l1.addWidget(self.__number)
        l1.addSpacing(35)
        l1.addWidget(self.__minimum)
        l1.addWidget(self.__slider)
        l1.addWidget(self.__maximum)
        component.setLayout(l1)
        return component

    @property
    def slider(self) -> QSlider:
        """
        Refernce to the slider object
        :return:
        """
        return self.__slider

    @property
    def number(self) -> float:
        """
        The value of the parameter
        :return:
        """
        return float(self.__number.text())

    @number.setter
    def number(self, value: int):
        self.__number.setText(f'{value / self.__divisor}')

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
        if float(self.maximum.text()) <= value / self.__divisor:
            raise Exception('Minimum cannot exceed maximum!')
        if self.__slider.value() <= value:
            self.__slider.setValue(value)
        self.__slider.setMinimum(value)
        self.__minimum.setText(f'{value / self.__divisor}')

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
        if float(self.minimum.text()) >= value / self.__divisor:
            raise Exception('Maximum cannot be lower than minimum!')
        if self.__slider.value() >= value:
            self.__slider.setValue(value)
        self.__slider.setMaximum(value)
        self.__maximum.setText(f'{value / self.__divisor}')

    def __change_slider(self, value: int):
        """
        Callback connected to the slider.valueChanged slot
        :param value:
        :return:
        """
        self.number = value


class Boolean(Parameter):
    def __init__(self, label_true: str, label_false: str, name: str = ''):
        self.__label_true = label_true
        self.__label_false = label_false
        setting.add_font_callback(self.adjust_fonts)
        super().__init__(name=name)

    def create_child(self) -> QWidget:
        component = QWidget(None)
        self.__button_true = QRadioButton(self.__label_true)
        self.__button_false = QRadioButton(self.__label_false)
        self.__button_true.clicked.connect(self.__change_button)
        self.__button_false.clicked.connect(self.__change_button)
        self.__status = False
        self.status = False

        l1 = QHBoxLayout()
        l1.addWidget(self.__button_true)
        l1.addWidget(self.__button_false)
        component.setLayout(l1)
        self.adjust_fonts()
        return component

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
        self.__button_false.setFont(font)
        self.__button_true.setFont(font)


class SingleSelect(Parameter):
    def __init__(self, labels: list[str], name: str = '') -> None:
        if not len(labels):
            raise Exception("No labels given")
        self.__labels = labels
        setting.add_font_callback(self.adjust_fonts)

        super().__init__(name=name)

    def create_child(self) -> QWidget:
        component = QWidget(None)
        self.__group = QButtonGroup(self.__component)
        layout = FlowLayout()

        for i, label in enumerate(self.__labels):
            radio = QRadioButton(label)
            self.__group.addButton(radio)
            layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)

        self.__status = self.__labels[0]
        self.__component.setLayout(layout)
        self.__group.buttonClicked.connect(self.__change_button)
        self.adjust_fonts()
        return component

    @property
    def status(self) -> str:
        return self.__status

    @status.setter
    def status(self, value: str) -> None:
        self.__status = value
        i: QRadioButton
        for i in self.__group.buttons():
            i.setChecked(i.text() == value)

    def __change_button(self, value: QRadioButton) -> None:
        self.__status = value.text()

    def adjust_fonts(self) -> None:
        font = setting.fonts[LabelLevel.P].generate_q()
        i: QRadioButton
        for i in self.__group.buttons():
            i.setFont(font)
