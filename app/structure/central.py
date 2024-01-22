from PyQt5.QtWidgets import QMainWindow, QTabWidget, QTabBar

from app.config import setting
from app.general.enums import LabelLevel
from app.structure.settings import Settings
from app.video.stream import StreamWidget


class CentralWidget(QTabWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent=parent)

        # Property Setup
        self.__stream = StreamWidget(self)
        self.__settings = Settings(self)

        # Widget Setup
        self.addTab(self.stream, 'Stream')
        self.addTab(self.settings, 'Settings')
        setting.add_font_callback(self.change_tab_font)
        self.change_tab_font()
        self.show()

    @property
    def stream(self) -> StreamWidget:
        return self.__stream

    @property
    def settings(self) -> Settings:
        return self.__settings

    def change_tab_font(self) -> None:
        font = setting.fonts[LabelLevel.H4].generate_q()
        bar: QTabBar = self.tabBar()
        bar.setFont(font)
