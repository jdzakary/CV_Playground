from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QListWidget


class CustomList(QListWidget):
    """
    A QListWidget with additional functionality.
    """
    def __init__(self, parent):
        super().__init__(parent=parent)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Override the key event.
        Add support for deletion
        :param event:
        :return:
        """
        if event.key() == Qt.Key_Delete:
            row = self.currentRow()
            if row > -1:
                self.takeItem(row)
                self.callback_remove(row)

    def callback_remove(self, index: int) -> None:
        """
        Callback triggered when an item is deleted from the list
        :param index:
        :return:
        """
        pass
