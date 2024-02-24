"""
This is the main program. It imports the base window and launches the application
"""
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app.base import BaseWindow

try:
    """
    Some files take a very long time to load using import_lib
    due to large libraries being imported in a secondary thread.
    
    These imports take much less time when imported
    as part of the main program. Consequently, users may wish
    to preemptively import some libraries here to expedite load time.
    
    For example, the YOLO class from ultralytics takes a long time
    to load using import_lib. However, importing it here only slightly
    increases GUI launch time.
    """
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Missing Some Optional Modules")

if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = '1'
    application = QApplication([])
    application.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = BaseWindow('CV Playground')
    sys.exit(application.exec())
