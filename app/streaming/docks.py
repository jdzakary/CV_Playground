from __future__ import annotations
from functools import partial
from typing import Type

from PyQt5.QtCore import Qt, QModelIndex, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QHBoxLayout, QTabWidget,
    QTabBar, QVBoxLayout,
    QAbstractItemView, QComboBox, QPushButton, QCompleter, QListWidgetItem, QGridLayout,
)

from app.config import setting
from app.data.general import signal_manager
from app.data.steaming import stream_operations
from app.general.enums import LabelLevel
from app.general.lists import CustomList
from app.general.spinner import Spinner
from app.general.text import Label
from app.streaming.operations import ModuleManager
from app.streaming.processing import Operation


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


class LoadTab(QWidget):
    """
    The tab for loading custom python modules into the program
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.__module_manager = ModuleManager(self)
        layout = self.__create_layout()
        self.__grid: QGridLayout = layout.children()[0]
        self.__dependencies: QGridLayout = layout.children()[1]
        self.setLayout(layout)
        signal_manager['modulesUpdated'].connect(self.__update)

    @property
    def module_manager(self) -> ModuleManager:
        return self.__module_manager

    def __create_layout(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        l1 = QGridLayout()

        for row, (file, custom) in enumerate(self.__module_manager.modules.items()):
            spinner = Spinner(self)
            spinner.setHidden(True)
            spinner.setAlignment(Qt.AlignHCenter)
            button = QPushButton('Load File')
            button.clicked.connect(partial(self.__start_loading, file))
            l1.addWidget(Label(file, LabelLevel.P), row, 0)
            l1.addWidget(button, row, 1)
            l1.addWidget(spinner, row, 2)
        l1.setAlignment(Qt.AlignTop)
        l1.setColumnStretch(0, 2)
        l1.setColumnStretch(0, 1)
        l2 = QGridLayout()

        layout.addWidget(Label('Import Files', LabelLevel.H3))
        layout.addLayout(l1)
        layout.addSpacing(30)
        layout.addWidget(Label('Missing Dependencies', LabelLevel.H3))
        layout.addLayout(l2)
        layout.setAlignment(Qt.AlignTop)
        return layout

    def __start_loading(self, file: str) -> None:
        self.__module_manager.attempt_import(file)
        self.__update()

    def __update(self) -> None:
        for row in range(0, self.__grid.rowCount()):
            label: Label = self.__grid.itemAtPosition(row, 0).widget()
            module = self.__module_manager.modules[label.text()]
            button: QPushButton = self.__grid.itemAtPosition(row, 1).widget()
            spinner: Spinner = self.__grid.itemAtPosition(row, 2).widget()
            spinner.setHidden(not module.importing)
            button.setDisabled(module.imported | module.importing)

        for file in self.__module_manager.modules.values():
            if len(file.dependencies):
                self.__dependencies.addWidget(Label(file.name, LabelLevel.P), 0, 0)
                self.__dependencies.addWidget(Label(','.join(file.dependencies), LabelLevel.P), 0, 1)


class OperationAdder(QThread):
    operationInitialized = pyqtSignal(Operation, name='operationInitialized')
    initializationError = pyqtSignal(Exception, name='initializationError')

    def __init__(self, parent, operation: Type[Operation]) -> None:
        super().__init__(parent)
        self.__opp_class = operation

    def run(self):
        try:
            opp = self.__opp_class()
            self.operationInitialized.emit(opp)
        except Exception as e:
            self.initializationError.emit(e)


class ManageOperators(QDockWidget):
    """
    Dockable tool window for managing the real-time cv operations
    applied to the stream.
    """

    def __init__(self, parent):
        super().__init__("Manage Stream Operators", parent)

        signal_manager['update_latency'].connect(self.__update_latency)
        signal_manager['modulesUpdated'].connect(self.__update_modules)

        self.__select_opp = QComboBox(self)
        self.__view_opp = CustomList(self)
        self.__view_opp.callback_remove = self.__remove_opp
        self.__tabs = QTabWidget(self)
        self.__load_tab = LoadTab(self)
        self.__add = QPushButton('Add Operation')
        self.__opp_loading = Spinner(self)
        self.__opp_loading.setHidden(True)

        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.__content = self.__create_content()
        self.update_fonts()
        self.setWidget(self.__content)

    def __create_content(self) -> QWidget:
        content = QWidget(self)
        configure = QWidget(content)
        configure.setLayout(QVBoxLayout())
        self.__tabs.addTab(self.__tab_select(), 'Select')
        self.__tabs.addTab(configure, 'Configure')
        self.__tabs.addTab(self.__load_tab, 'Load')
        self.change_tab_font(self.__tabs)
        setting.add_font_callback(lambda: self.change_tab_font(self.__tabs))

        layout = QVBoxLayout()
        layout.addWidget(Label('Manage Operations:', LabelLevel.H1))
        layout.addWidget(self.__tabs)
        layout.setAlignment(Qt.AlignTop)
        content.setLayout(layout)

        return content

    @staticmethod
    def change_tab_font(tabs: QTabWidget) -> None:
        font = setting.fonts[LabelLevel.H4].generate_q()
        bar: QTabBar = tabs.tabBar()
        bar.setFont(font)

    def __tab_select(self) -> QWidget:
        widget = QWidget(self)

        # Combo Box to select operation
        self.__select_opp.setInsertPolicy(QComboBox.NoInsert)
        self.__select_opp.setEditable(True)
        self.__select_opp.completer().setCompletionMode(QCompleter.PopupCompletion)

        # Button to add operation to end of list
        self.__add.clicked.connect(self.__add_opp)

        # Layout the controls
        l2 = QHBoxLayout()
        l2.addWidget(self.__select_opp)
        l2.addWidget(self.__add)
        l2.addWidget(self.__opp_loading)
        l2.setStretch(0, 2)
        l2.setStretch(1, 1)

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
        moved = stream_operations.pop(row_start)
        if row_dest == 0:
            stream_operations.insert(0, moved)
        else:
            stream_operations.insert(row_dest - 1, moved)
        self.__tab_configure()

    def __add_opp(self):
        idx = self.__select_opp.currentIndex()
        name = self.__select_opp.itemText(idx)
        opp_class = self.__load_tab.module_manager.operations[name]
        if opp_class.InitializeInSeparateThread:
            self.__add.setDisabled(True)
            self.__opp_loading.setHidden(False)
            worker = OperationAdder(self, opp_class)
            worker.operationInitialized.connect(self.__handle_new_opp)
            worker.initializationError.connect(self.__handle_init_error)
            worker.start()
        else:
            opp = opp_class()
            opp.create_controls()
            proxy = QListWidgetItem()
            widget = OperatorListItem(name)
            proxy.setSizeHint(widget.sizeHint())
            self.__view_opp.addItem(proxy)
            self.__view_opp.setItemWidget(proxy, widget)
            stream_operations.append(opp)
            self.__tab_configure()

    def __remove_opp(self, index: int):
        stream_operations.pop(index)
        self.__tab_configure()

    def __tab_configure(self) -> None:
        """
        Every time
        :return:
        """
        l1 = QVBoxLayout()
        for i in stream_operations:
            l1.addWidget(Label(i.name, LabelLevel.H3))
            for p in i.params:
                l1.addWidget(p.component)
        l1.setAlignment(Qt.AlignTop)
        tab = self.__tabs.widget(1)
        QWidget(None).setLayout(tab.layout())
        tab.setLayout(l1)

    def update_fonts(self) -> None:
        p = setting.fonts[LabelLevel.P].generate_q()
        self.__select_opp.setFont(p)
        self.__view_opp.setFont(p)

    def __update_latency(self, latency: list[float]) -> None:
        for i in range(self.__view_opp.count()):
            try:
                item = self.__view_opp.item(i)
                widget: OperatorListItem = self.__view_opp.itemWidget(item)
                widget.update_latency(latency[i])
            except Exception as e:
                print(e)

    def __update_modules(self) -> None:
        existing = [self.__select_opp.itemText(x) for x in range(self.__select_opp.count())]
        for opp in self.__load_tab.module_manager.operations:
            if opp not in existing:
                self.__select_opp.addItem(opp)

    def __handle_new_opp(self, opp: Operation) -> None:
        opp.create_controls()
        self.__add.setEnabled(True)
        self.__opp_loading.setHidden(True)
        proxy = QListWidgetItem()
        widget = OperatorListItem(opp.name)
        proxy.setSizeHint(widget.sizeHint())
        self.__view_opp.addItem(proxy)
        self.__view_opp.setItemWidget(proxy, widget)
        stream_operations.append(opp)
        self.__tab_configure()

    def __handle_init_error(self, e: Exception) -> None:
        print(f'An error occurred! {e}')
        self.__add.setEnabled(True)
        self.__opp_loading.setHidden(True)
