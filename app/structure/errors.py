import traceback

from PyQt5.QtWidgets import QMessageBox, QGridLayout, QSpacerItem, QSizePolicy


def error_report(e: Exception, description: str) -> None:
    message = QMessageBox()
    message.setText(description)
    message.setInformativeText(str(e))
    message.setDetailedText('\n'.join(traceback.format_tb(e.__traceback__)))
    message.setWindowTitle('Error')
    message.setIcon(QMessageBox.Critical)
    message.setStandardButtons(QMessageBox.Ok)
    layout: QGridLayout = message.layout()
    spacer = QSpacerItem(1000, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
    layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
    message.exec()
