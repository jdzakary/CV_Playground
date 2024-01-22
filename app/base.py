from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction

from app.config import setting
from app.structure.central import CentralWidget
from app.structure.docks import ManageOperators


class BaseWindow(QMainWindow):
    def __init__(self, title: str):
        super().__init__(parent=None)

        # Properties Setup
        self.__central_widget = CentralWidget(self)
        self.__menu_bar = self.__create_menu()
        self.__manage_ops = ManageOperators(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.manage_ops)

        # Window Setup
        self.manage_ops.update_operations = self.central_widget.stream.process.change_operations
        self.setWindowTitle(title)
        self.setCentralWidget(self.central_widget)
        self.showMaximized()

    @property
    def central_widget(self) -> CentralWidget:
        return self.__central_widget

    @property
    def menu_bar(self) -> QMenuBar:
        return self.__menu_bar

    @property
    def manage_ops(self) -> ManageOperators:
        return self.__manage_ops

    def __create_menu(self) -> QMenuBar:
        menu_bar = QMenuBar(self)

        tools: QMenu = menu_bar.addMenu('Tool Windows')
        action1 = QAction('Manage Operators', tools)
        action1.triggered.connect(lambda: self.manage_ops.setHidden(False))
        tools.addAction(action1)

        self.setMenuBar(menu_bar)
        return menu_bar

    def closeEvent(self, a0, QCloseEvent=None):
        setting.dump_settings()
