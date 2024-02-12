import importlib

from app.streaming.processing import Operation
import os

for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    module = importlib.import_module(f'app.streaming.operations.{module[:-3]}')
del module

OPP_MAP = {cls.name: cls for cls in Operation.__subclasses__()}
