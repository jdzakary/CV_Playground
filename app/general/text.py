from PyQt5.QtWidgets import QLabel

from app.general.enums import LabelLevel
from app.config import setting


class Label(QLabel):
    """
    A text widget that supports automatic re-formatting
    whenever a global setting is changed.
    """
    def __init__(self, text: str, level: LabelLevel):
        super().__init__(text)

        # Property Setup
        self.__level = level

        # Widget Setup
        self.update_font()
        setting.add_font_callback(self.update_font)

    @property
    def level(self) -> LabelLevel:
        """
        What kind of text is this?
        :return:
        """
        return self.__level

    @level.setter
    def level(self, value: LabelLevel):
        """
        Change the text type
        :param value:
        :return:
        """
        if not isinstance(value, LabelLevel):
            raise Exception('Invalid Font Level')
        self.__level = value

    def update_font(self) -> None:
        """
        Callback for updating font
        :return:
        """
        font = setting.fonts[self.level].generate_q()
        try:
            self.setFont(font)
        except RuntimeError:
            setting.remove_font_callback(self.update_font)
