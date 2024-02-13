from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction

from app.config import setting
from app.structure.central import CentralWidget
from app.streaming.docks import ManageOperators


class BaseWindow(QMainWindow):
    """
    The main window presented by the application
    """
    def __init__(self, title: str):
        super().__init__(parent=None)

        # Properties Setup
        self.__central_widget = CentralWidget(self)
        self.__menu_bar = self.__create_menu()
        self.__manage_ops = ManageOperators(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.manage_ops)

        # Window Setup
        self.setWindowTitle(title)
        self.setCentralWidget(self.central_widget)
        self.showMaximized()

    @property
    def central_widget(self) -> CentralWidget:
        """
        Reference to the central widget
        :return:
        """
        return self.__central_widget

    @property
    def menu_bar(self) -> QMenuBar:
        """
        Reference to the menubar
        :return:
        """
        return self.__menu_bar

    @property
    def manage_ops(self) -> ManageOperators:
        """
        Reference to the tool window for managing real-time operators
        :return:
        """
        return self.__manage_ops

    def __create_menu(self) -> QMenuBar:
        """
        Create the menubar
        :return:
        """
        menu_bar = QMenuBar(self)

        tools: QMenu = menu_bar.addMenu('Tool Windows')
        action1 = QAction('Manage Operators', tools)
        action1.triggered.connect(lambda: self.manage_ops.setHidden(False))
        tools.addAction(action1)

        self.setMenuBar(menu_bar)
        return menu_bar

    def closeEvent(self, a0, QCloseEvent=None):
        """
        Handle the application closing.
        :param a0:
        :param QCloseEvent:
        :return:
        """
        setting.dump_settings()
        self.central_widget.stream.capture.exit()
        self.central_widget.stream.process.exit()
