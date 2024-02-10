from collections import deque

import numpy as np


class SimpleAvg:
    def __init__(self, name: str, lookback: int):
        self.__name = name
        self.__data = deque([1.0]*lookback, maxlen=lookback)

    @property
    def name(self) -> str:
        return self.__name

    def current_value(self) -> float:
        return np.mean(self.__data)

    def update(self, number: float) -> None:
        self.__data.appendleft(number)
        self.__data.pop()
