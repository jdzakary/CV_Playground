from PyQt5.QtCore import QSize
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QLabel


class LoadingSpinner(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__movie = QMovie("assets/loading.gif")
        self.__movie.setScaledSize(QSize(20, 20))
        self.setMovie(self.__movie)
        self.start_animation()

    def start_animation(self):
        self.__movie.start()

    def stop_animation(self):
        self.__movie.stop()
