from PyQt5.QtWidgets import QApplication

from app.base import BaseWindow

if __name__ == '__main__':
    application = QApplication([])
    window = BaseWindow('CV Playground')
    application.exec()
