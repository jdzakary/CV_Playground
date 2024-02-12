from __future__ import annotations

from typing import Callable, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PyQt5.QtCore import pyqtBoundSignal


class FakeSignal:
    """
    An object to be returned if a signal has not yet been added
    to the signal manager. Provides a ".connect" method that "fools"
    the caller into thinking the slot has been connected to a signal.

    The slot is added to the waiting que and will be connected if/when
    the signal with the correct name is added to the signal manager
    """
    def __init__(self, manager: SignalManager, key: str) -> None:
        self.__manager = manager
        self.__key = key

    def connect(self, slot: Any) -> None:
        if self.__key not in self.__manager.waiting:
            self.__manager.waiting[self.__key] = [slot]
        else:
            self.__manager.waiting[self.__key].append(slot)


class SignalManager(dict):
    """
    Special dictionary for managing PyQt Signals
    and connecting them to the appropriate slots.
    Solves the issue of connecting a slot to a signal that
    has not been created yet due to the application build order.

    If signals['my_signal'] is defined, it returns a pyqtBoundSignal.
    If signals['my_signal'] is not defined, it returns a FakeSignal instance.
    A FakeSignal instance allows the user to call .connect and put
    a slot on a waiting stack.

    When a signal is added, if any waiting slots will be connected.
    """
    def __init__(self):
        super().__init__()
        self.__waiting: dict[str, list] = {}

    def __setitem__(self, key: str, value: pyqtBoundSignal) -> None:
        super().__setitem__(key, value)
        for i in self.__waiting.get(key, []):
            self[key].connect(i)
        self.__waiting[key] = []

    def __getitem__(self, key: str) -> pyqtBoundSignal | FakeSignal:
        try:
            return super().__getitem__(key)
        except KeyError:
            return FakeSignal(self, key)

    @property
    def waiting(self) -> dict[str, list]:
        return self.__waiting


signal_manager = SignalManager()


class DataManager:
    def __init__(self, mapping: dict) -> None:
        self.__mapping = mapping
        for key, value in self.__mapping.items():
            signal_manager[key].connect(value)

    @property
    def mapping(self) -> dict[str, Callable]:
        return self.__mapping

    def replace_listener(self, key: str, listener: Callable) -> None:
        signal_manager[key].disconnect(self.__mapping[key])
        signal_manager[key].connect(listener)
