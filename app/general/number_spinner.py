from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtWidgets import QSpinBox

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QLineEdit
    from PyQt5.QtGui import QKeyEvent


class CustomSpinBox(QSpinBox):
    newValue = pyqtSignal(int, name='newValue')

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            line_edit: QLineEdit = self.lineEdit()
            line_edit.clearFocus()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, *args, **kwargs):
        super().focusOutEvent(*args, **kwargs)
        self.newValue.emit(self.value())
