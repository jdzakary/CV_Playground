from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QSpinBox, QGridLayout, QVBoxLayout, QFontDialog

from app.config import setting
from app.general.enums import LabelLevel
from app.general.text import Label


class Settings(QWidget):
    """
    Main widget for the settings tab
    """
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.__setup()

    def __fonts(self) -> QGridLayout:
        """
        Settings related to text font and style
        :return:
        """
        layout = QGridLayout()
        for i, level in enumerate(LabelLevel):
            # Font Size Selector
            spin = QSpinBox(self)
            callback = partial(self.change_font_size, level)
            spin.setValue(setting.fonts[level].point_size)
            spin.valueChanged.connect(callback)

            # Setup Row
            layout.addWidget(Label(level.value, level), i, 0, 1, 1)
            layout.addWidget(spin, i, 1, 1, 1)

        layout.setAlignment(Qt.AlignCenter)
        layout.setColumnMinimumWidth(1, 50)
        return layout

    def __setup(self):
        """
        Overall widget setup
        :return:
        """
        layout = QVBoxLayout()
        layout.addWidget(Label('Adjust Text:', LabelLevel.H1))
        layout.addLayout(self.__fonts())
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    @staticmethod
    def change_font_size(level: LabelLevel, value: int):
        """
        Callback for changing font size
        :param level:
        :param value:
        :return:
        """
        setting.fonts[level].point_size = value
