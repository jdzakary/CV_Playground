import json
import os
from typing import Callable
from PyQt5.QtGui import QFont

from app.general.enums import LabelLevel


class Font:
    """
    A font settings object.
    Controls font family and point size for various text levels
    """
    def __init__(self, point_size: int, family: str, callback: Callable):
        self.__point_size = point_size
        self.__family = family
        self.__callback = callback

    @property
    def point_size(self) -> int:
        """
        Size of the font
        :return:
        """
        return self.__point_size

    @point_size.setter
    def point_size(self, value: int):
        """
        Change the point size and force text to be updated throughout app
        :param value:
        :return:
        """
        self.__point_size = value
        self.__callback()

    @property
    def family(self) -> str:
        """
        Family of the font
        :return:
        """
        return self.__family

    @family.setter
    def family(self, value: str):
        """
        Change the family and force text to be updated throughout the app
        :param value:
        :return:
        """
        self.__family = value
        self.__callback()

    def generate_q(self) -> QFont:
        """
        Generate the QFont object from the saved attributes.
        This QFont object is applied to QLabels and other textual items
        :return:
        """
        font = QFont()
        font.setFamily(self.__family)
        font.setPointSize(self.__point_size)
        return font

    def to_json(self) -> dict:
        """
        Convert this object to a json representation for persistence
        :return:
        """
        return dict(
            point_size=self.point_size,
            family=self.family
        )


class Settings:
    """
    Settings class accessible from everywhere in the app
    """
    def __init__(self):
        data = self.load_settings()
        # Property Setup
        self.__fonts: dict[LabelLevel, Font] = {
            LabelLevel(k): Font(**v, callback=self.update_font) for (k, v) in data['fonts'].items()
        }
        self.__font_callbacks: list[Callable] = []

    @property
    def fonts(self) -> dict[LabelLevel, Font]:
        """
        Font setting objects
        :return:
        """
        return self.__fonts

    def add_font_callback(self, callback: Callable) -> None:
        """
        Add a callback to execute whenever the user changes a font setting
        :param callback:
        :return:
        """
        self.__font_callbacks.append(callback)

    def remove_font_callback(self, callback: Callable) -> None:
        self.__font_callbacks.remove(callback)

    def update_font(self):
        """
        Execute callbacks when the user changes a font setting
        :return:
        """
        for i in self.__font_callbacks:
            i()

    @staticmethod
    def load_settings() -> dict:
        """
        Load settings from disk
        :return:
        """
        print(os.getcwd())
        with open('app/settings.json', 'r') as file:
            return json.load(file)

    def dump_settings(self):
        """
        Dump settings to file when the user closes the program
        :return:
        """
        data = dict(
            fonts={k.value: v.to_json() for (k, v) in self.fonts.items()}
        )
        with open('app/settings.json', 'w') as file:
            json.dump(data, file, indent=2)


# Global settings singleton
setting = Settings()
