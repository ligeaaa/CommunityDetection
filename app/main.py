import datetime
import os
import sys

from loguru import logger
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from app.ui.main import Ui_Dialog
from app.utils.qt_main_window import RQMainWindow


class Controller(RQMainWindow):
    def __init__(self, ui: Ui_Dialog):
        super().__init__(None, close_call=None)
        self.ui = ui
        self.init_event()

    def init_event(self):
        self.ui.button_process.clicked.connect(self.test)
        self.ui.list_dataset.currentIndexChanged.connect(self.display_raw_dataset)

    def test(self):
        self.ui.text_choose_algorithm.setText("123123123")

    def display_raw_dataset(self):
        dataset_name = self.ui.list_dataset.currentText()
        self.ui.text_choose_algorithm.setText(dataset_name)


def start():
    import logging

    LOGFORMAT = " %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"

    sys._excepthook = sys.excepthook
    logging.basicConfig(level=logging.DEBUG, format=LOGFORMAT, datefmt="%H:%M:%S")
    logformat = "{time:YYYY-MM-DD HH:mm:ss} | {process} | {thread} | {level} | {file}:{line} - {message}"

    logger.add(
        os.path.join(
            "c:/tmp/",
            f'community-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log',
        ),
        format=logformat,
        rotation="500 MB",
        level=logging.DEBUG,
    )

    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)

    MainWindow = RQMainWindow()

    ui = Ui_Dialog()
    ui.setupUi(MainWindow)

    # c = Controller(ui)

    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start()
