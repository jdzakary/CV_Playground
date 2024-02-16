from __future__ import annotations

import os
from importlib import import_module
from inspect import isclass
from typing import Type

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from app.data.general import signal_manager
from app.streaming.processing import Operation


FILE_NAMES = [
    x[:-3] for x in os.listdir(os.path.dirname(__file__))
    if x[-3:] == '.py' and x != '__init__.py'
]


class ModuleImporter(QThread):
    moduleImported = pyqtSignal(str, list, name='moduleImported')
    importError = pyqtSignal(str, Exception, name='importError')

    def __init__(self, module_name: str, parent):
        super().__init__(parent)
        self.__module_name = module_name

    def run(self) -> None:
        result = []
        # noinspection PyBroadException
        try:
            module = import_module(f'app.streaming.operations.{self.__module_name}')
            for value in module.__dict__.values():
                if isclass(value) and issubclass(value, Operation) and value != Operation:
                    result.append(value)
            self.moduleImported.emit(self.__module_name, result)
        except Exception as e:
            self.importError.emit(self.__module_name, e)


class ModuleManager(QObject):
    modulesUpdated = pyqtSignal(name='modulesUpdated')

    def __init__(self, parent):
        super().__init__(parent)
        signal_manager['modulesUpdated'] = self.modulesUpdated
        self.__modules = {x: CustomModule(x) for x in FILE_NAMES}
        self.__operations = {}

    @property
    def modules(self) -> dict[str, CustomModule]:
        return self.__modules

    @property
    def operations(self) -> dict[str, Type[Operation]]:
        # TODO: Change dock window so operations are visualized by module, not all in one list
        return self.__operations

    def attempt_import(self, module_name: str) -> None:
        self.__modules[module_name].importing = True
        worker = ModuleImporter(module_name, self)
        worker.moduleImported.connect(self.__handle_success)
        worker.importError.connect(self.__handle_error)
        worker.start()

    def __handle_success(self, module_name: str, result: list[Type[Operation]]) -> None:
        self.__modules[module_name].importing = False
        for i in result:
            self.__operations[i.name] = i
        self.__modules[module_name].imported = True
        self.modulesUpdated.emit()

    def __handle_error(self, module_name: str, e: Exception) -> None:
        self.__modules[module_name].importing = False
        print(e)
        if isinstance(e, ModuleNotFoundError):
            self.__modules[module_name].dependencies.append(e.name)
        self.modulesUpdated.emit()


class CustomModule:
    def __init__(self, name: str):
        self.__name = name
        self.__imported = False
        self.__importing = False
        self.__dependencies = []
        self.__operations = []

    @property
    def name(self) -> str:
        return self.__name

    @property
    def imported(self) -> bool:
        return self.__imported

    @property
    def importing(self) -> bool:
        return self.__importing

    @property
    def dependencies(self) -> list[str]:
        return self.__dependencies

    @property
    def operations(self) -> list[Type[Operation]]:
        return self.__operations

    @imported.setter
    def imported(self, value: bool) -> None:
        self.__imported = value

    @importing.setter
    def importing(self, value: bool) -> None:
        self.__importing = value

