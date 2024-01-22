import importlib

from app.video.processing import Operation
import os

for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    module = importlib.import_module(f'app.video.operations.{module[:-3]}')
del module

OPP_MAP = {cls.name: cls for cls in Operation.__subclasses__()}
