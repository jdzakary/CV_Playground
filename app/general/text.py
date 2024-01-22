from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel

from app.general.enums import LabelLevel
from app.config import setting


class Label(QLabel):
    def __init__(self, text: str, level: LabelLevel):
        super().__init__(text)

        # Property Setup
        self.__level = level

        # Widget Setup
        self.update_font()
        setting.add_font_callback(self.update_font)

    @property
    def level(self) -> LabelLevel:
        return self.__level

    @level.setter
    def level(self, value: LabelLevel):
        if not isinstance(value, LabelLevel):
            raise Exception('Invalid Font Level')
        self.__level = value

    def update_font(self) -> None:
        font = setting.fonts[self.level].generate_q()
        self.setFont(font)
