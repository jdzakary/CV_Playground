from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QWidget, QHBoxLayout, QButtonGroup, QRadioButton, QVBoxLayout, QSpinBox

from app.config import setting
from app.general.enums import LabelLevel
from app.general.layouts import FlowLayout
from app.general.number_spinner import CustomSpinBox
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
        self.__child_functions = []

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
        for i in self.__child_functions:
            i()

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

    @property
    def child_functions(self) -> list:
        """
        Used to link parameters together.
        For example, the two sliders in a canny edge detector are linked
        such that the value of the slider threshold is the minimum of the upper slider
        :return:
        """
        return self.__child_functions


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


class NewSlider(Parameter):
    def __init__(
        self,
        maximum: float,
        minimum: float,
        step: float = 1,
        default: float = None,
        name: str = '',
        label_precision: int = 0,
    ):
        self.__minimum = minimum
        self.__maximum = maximum
        self.__step = step
        self.__number = minimum
        self.__default = default
        self.__label_precision = label_precision

        super().__init__(name=name)

    @property
    def slider(self) -> QSlider:
        return self.__slider

    @property
    def number(self) -> float:
        return self.__number

    @number.setter
    def number(self, value: float):
        if self.__maximum < value:
            raise Exception('Cannot set number above maximum!')
        if self.__minimum > value:
            raise Exception('Cannot set number below minimum!')
        self.__number = value
        self.__slider.setValue(self.__number_to_slider(value))

    @property
    def minimum(self) -> float:
        return self.__minimum

    @minimum.setter
    def minimum(self, value: float):
        self.__minimum = value
        start, stop = self.__calculate_endpoints()
        self.__slider.setRange(start, stop)
        self.__update_labels()
        if value > self.__number:
            self.number = value

    @property
    def maximum(self) -> float:
        return self.__maximum

    @maximum.setter
    def maximum(self, value: float):
        self.__maximum = value
        start, stop = self.__calculate_endpoints()
        self.__slider.setRange(start, stop)
        self.__update_labels()
        if value < self.__number:
            self.number = value

    def __calculate_endpoints(self) -> tuple[int, int]:
        steps = (self.__maximum - self.__minimum) / self.__step
        return 1, int(1 + steps//1)

    def __slider_changed(self, value: int) -> None:
        self.__number = self.__slider_to_number(value)
        self.__update_labels()

    def __update_labels(self) -> None:
        self.__label_max.setText(f'{self.__maximum:.{self.__label_precision}f}')
        self.__label_number.setText(f'{self.__number:.{self.__label_precision}f}')
        self.__label_min.setText(f'{self.__minimum:.{self.__label_precision}f}')

    def __slider_to_number(self, value: int) -> float:
        start, stop = self.__calculate_endpoints()
        return (value - start) * self.__step + self.__minimum

    def __number_to_slider(self, value: float) -> int:
        start, stop = self.__calculate_endpoints()
        return int((value - self.__minimum) / self.__step) + start

    def create_child(self) -> QWidget:
        start, stop = self.__calculate_endpoints()
        self.__slider = QSlider()
        self.__slider.setRange(start, stop)
        self.__slider.setValue(start)
        self.__slider.valueChanged.connect(self.__slider_changed)
        self.__slider.setOrientation(Qt.Horizontal)

        component = QWidget(None)
        self.__label_number = Label(f'{self.__number:.{self.__label_precision}f}', LabelLevel.P)
        self.__label_min = Label(f'{self.__minimum:.{self.__label_precision}f}', LabelLevel.P)
        self.__label_max = Label(f'{self.__maximum:.{self.__label_precision}f}', LabelLevel.P)

        if self.__default is not None:
            self.number = self.__default

        l1 = QHBoxLayout()
        l1.addWidget(Label('Current Value:', LabelLevel.P))
        l1.addWidget(self.__label_number)
        l1.addSpacing(35)
        l1.addWidget(self.__label_min)
        l1.addWidget(self.__slider)
        l1.addWidget(self.__label_max)
        component.setLayout(l1)
        return component


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


class IntegerEntry(Parameter):
    def __init__(
        self,
        name: str = '',
        min_value: int = None,
        max_value: int = None,
        step: int = None,
        default: int = None,
    ):
        self.__min_value = min_value
        self.__max_value = max_value
        self.__step = step
        self.__default = default
        super().__init__(name)

    def create_child(self) -> QWidget:
        component = QWidget(None)
        self.__spinner = CustomSpinBox()

        if self.__min_value is not None:
            self.__spinner.setMinimum(self.__min_value)
        if self.__max_value is not None:
            self.__spinner.setMaximum(self.__max_value)
        if self.__step is not None:
            self.__spinner.setSingleStep(self.__step)
        if self.__default is not None:
            self.__spinner.setValue(self.__default)

        l1 = QHBoxLayout()
        l1.addWidget(Label('Enter Value', LabelLevel.P))
        l1.addWidget(self.__spinner)
        component.setLayout(l1)
        return component

    @property
    def spin_box(self) -> CustomSpinBox:
        return self.__spinner

    @property
    def number(self) -> int:
        return self.__spinner.value()


class SingleSelect(Parameter):
    def __init__(self, labels: list[str], name: str = '') -> None:
        if not len(labels):
            raise Exception("No labels given")
        self.__labels = labels
        setting.add_font_callback(self.adjust_fonts)

        super().__init__(name=name)

    def create_child(self) -> QWidget:
        component = QWidget(None)
        self.__group = QButtonGroup(component)
        layout = FlowLayout()

        for i, label in enumerate(self.__labels):
            radio = QRadioButton(label)
            self.__group.addButton(radio)
            layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)

        self.__status = self.__labels[0]
        component.setLayout(layout)
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
