import json
import os
from typing import Callable
from PyQt5.QtGui import QFont

from app.general.enums import LabelLevel


class Font:
    def __init__(self, point_size: int, family: str, callback: Callable):
        self.__point_size = point_size
        self.__family = family
        self.__callback = callback

    @property
    def point_size(self) -> int:
        return self.__point_size

    @point_size.setter
    def point_size(self, value: int):
        self.__point_size = value
        self.__callback()

    @property
    def family(self) -> str:
        return self.__family

    @family.setter
    def family(self, value: str):
        self.__family = value
        self.__callback()

    def generate_q(self) -> QFont:
        font = QFont()
        font.setFamily(self.__family)
        font.setPointSize(self.__point_size)
        return font

    def to_json(self) -> dict:
        return dict(
            point_size=self.point_size,
            family=self.family
        )


class Settings:
    def __init__(self):
        data = self.load_settings()
        # Property Setup
        self.__fonts: dict[LabelLevel, Font] = {
            LabelLevel(k): Font(**v, callback=self.update_font) for (k, v) in data['fonts'].items()
        }
        self.__font_callbacks: list[Callable] = []

    @property
    def fonts(self) -> dict[LabelLevel, Font]:
        return self.__fonts

    def add_font_callback(self, callback: Callable) -> None:
        self.__font_callbacks.append(callback)

    def update_font(self):
        for i in self.__font_callbacks:
            i()

    @staticmethod
    def load_settings() -> dict:
        print(os.getcwd())
        with open('app/settings.json', 'r') as file:
            return json.load(file)

    def dump_settings(self):
        data = dict(
            fonts={k.value: v.to_json() for (k, v) in self.fonts.items()}
        )
        with open('app/settings.json', 'w') as file:
            json.dump(data, file, indent=2)


setting = Settings()
