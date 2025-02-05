import sys

import loguru
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QMainWindow, QMessageBox


class RQMainWindow(QMainWindow):
    def __init__(self, parent=None, close_call=None):
        self.close_call = close_call
        super(RQMainWindow, self).__init__(parent)

        # def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:

    #     loguru.logger.debug(f"{a0.key()} {a0.nativeScanCode()} {a0.spontaneous()}")
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "提示",
            "是否要关闭所有窗口?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        # reply.setWindowFlags(Qt.FramelessWindowHint)
        if reply == QMessageBox.Yes:
            try:
                event.accept()
                if self.close_call:
                    self.close_call()
            except Exception as e:
                loguru.logger.exception(e)
            sys.exit(0)  # 退出程序
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.isFullScreen():
            loguru.logger.debug("!fullScreen")
            self.showNormal()
        else:
            loguru.logger.debug("fullScreen")
            self.showFullScreen()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))
