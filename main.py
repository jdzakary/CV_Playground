"""
This is the main program. It imports the base window and launches the application
"""

from PyQt5.QtWidgets import QApplication

from app.base import BaseWindow

if __name__ == '__main__':
    application = QApplication([])
    window = BaseWindow('CV Playground')
    application.exec()
