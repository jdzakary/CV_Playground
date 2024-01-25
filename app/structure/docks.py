import numpy as np
from PyQt5.QtCore import Qt, QModelIndex, pyqtSlot
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QHBoxLayout, QTabWidget,
    QTabBar, QVBoxLayout,
    QAbstractItemView, QComboBox, QPushButton, QCompleter, QListWidgetItem, QListView, QLayout
)

from app.config import setting
from app.general.enums import LabelLevel
from app.general.lists import CustomList
from app.general.text import Label
from app.video.processing import Operation
from app.video.operations import OPP_MAP


class OperatorListItem(QWidget):
    def __init__(self, name: str):
        super().__init__(None)
        self.__name = Label(name, LabelLevel.P)
        self.__latency = Label('0 ms', LabelLevel.P)
        layout = QHBoxLayout()
        layout.addWidget(self.__name)
        layout.addStretch(1)
        layout.addWidget(self.__latency)
        layout.setContentsMargins(10, 0, 10, 0)
        self.setLayout(layout)

    def update_latency(self, latency: float) -> None:
        self.__latency.setText(f'{latency:.3f} ms')


class ManageOperators(QDockWidget):
    """
    Dockable tool window for managing the real-time cv operations
    applied to the video stream.
    """
    def __init__(self, parent):
        super().__init__("Manage Video Operators", parent)

        self.__operations: list[Operation] = []
        self.__select_opp = QComboBox(self)
        self.__view_opp = CustomList(self)
        self.__view_opp.callback_remove = self.__remove_opp
        self.__tabs = QTabWidget(self)

        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.__content = self.__create_content()
        self.update_fonts()
        # setting.add_font_callback(self.update_fonts)
        self.setWidget(self.content)

    @property
    def content(self) -> QWidget:
        return self.__content

    @property
    def operations(self) -> list[Operation]:
        return self.__operations

    @property
    def select_opp(self) -> QComboBox:
        return self.__select_opp

    def __create_content(self) -> QWidget:
        content = QWidget(self)
        configure = QWidget(content)
        configure.setLayout(QVBoxLayout())
        self.__tabs.addTab(self.__tab_select(), 'Select and Arrange')
        self.__tabs.addTab(configure, 'Configure Options')
        self.change_tab_font(self.__tabs)
        setting.add_font_callback(lambda: self.change_tab_font(self.__tabs))

        layout = QVBoxLayout()
        layout.addWidget(Label('Manage Operations:', LabelLevel.H1))
        layout.addWidget(self.__tabs)
        layout.setAlignment(Qt.AlignTop)
        content.setLayout(layout)

        return content

    @staticmethod
    def update_operations(ops: list[Operation]):
        pass

    @staticmethod
    def change_tab_font(tabs: QTabWidget) -> None:
        font = setting.fonts[LabelLevel.H4].generate_q()
        bar: QTabBar = tabs.tabBar()
        bar.setFont(font)

    def __tab_select(self) -> QWidget:
        widget = QWidget(self)

        # Combo Box to select operation
        self.select_opp.addItems(OPP_MAP.keys())
        self.select_opp.setInsertPolicy(QComboBox.NoInsert)
        self.select_opp.setEditable(True)
        self.select_opp.completer().setCompletionMode(QCompleter.PopupCompletion)

        # Button to add operation to end of list
        add = QPushButton('Add Operation')
        add.clicked.connect(self.__add_opp)

        # Layout the controls
        l2 = QHBoxLayout()
        l2.addWidget(self.select_opp)
        l2.addWidget(add)

        # List of operations
        self.__view_opp.setDragDropMode(QAbstractItemView.InternalMove)
        self.__view_opp.model().rowsMoved.connect(self.__order_changed)

        l1 = QVBoxLayout()
        l1.addLayout(l2)
        l1.addWidget(self.__view_opp)
        widget.setLayout(l1)

        return widget

    def __order_changed(
        self,
        idx1: QModelIndex,
        row_start: int,
        row_end: int,
        idx2: QModelIndex,
        row_dest: int
    ):
        moved = self.operations.pop(row_start)
        if row_dest == 0:
            self.operations.insert(0, moved)
        else:
            self.operations.insert(row_dest - 1, moved)
        self.__update_operations()

    def __add_opp(self):
        idx = self.select_opp.currentIndex()
        name = self.select_opp.itemText(idx)
        proxy = QListWidgetItem()
        widget = OperatorListItem(name)
        proxy.setSizeHint(widget.sizeHint())

        self.__view_opp.addItem(proxy)
        self.__view_opp.setItemWidget(proxy, widget)
        self.operations.append(OPP_MAP[name]())
        self.__update_operations()

    def __remove_opp(self, index: int):
        self.operations.pop(index)
        self.__update_operations()

    def __update_operations(self):
        self.update_operations(self.operations)
        self.__tab_configure()

    def __tab_configure(self) -> None:
        l1 = QVBoxLayout()
        for i in self.operations:
            l1.addWidget(Label(i.name, LabelLevel.H3))
            for p in i.params:
                l1.addWidget(p.component)
        l1.setAlignment(Qt.AlignTop)
        tab = self.__tabs.widget(1)
        QWidget(None).setLayout(tab.layout())
        tab.setLayout(l1)

    def update_fonts(self) -> None:
        p = setting.fonts[LabelLevel.P].generate_q()
        self.select_opp.setFont(p)
        self.__view_opp.setFont(p)

    @pyqtSlot(list, name='update_latency')
    def update_latency(self, latency) -> None:
        for i in range(self.__view_opp.count()):
            try:
                item = self.__view_opp.item(i)
                widget: OperatorListItem = self.__view_opp.itemWidget(item)
                widget.update_latency(latency[i])
            except Exception as e:
                print(e)
