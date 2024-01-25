import numpy as np


class SimpleAvg:
    def __init__(self, name: str, lookback: int):
        self.__name = name
        self.__data = np.ones((lookback,))

    @property
    def name(self) -> str:
        return self.__name

    def current_value(self) -> float:
        return self.__data.mean()

    def update(self, number: float) -> None:
        self.__data = np.roll(self.__data, 1)
        self.__data[0] = number
