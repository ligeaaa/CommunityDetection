import glob
import os.path
import typing
from pathlib import Path

import loguru


class QssLoader:
    def __init__(self):
        self._qss_dict: typing.Dict[str, str] = None
        self._qss_dir = os.path.join(os.path.dirname(__file__), "./qss")
        self.init()

    def init(self):
        self._qss_dict = {}
        for each in glob.glob(f"{self._qss_dir}/*.qss"):
            self._qss_dict[os.path.basename(each).removesuffix(".qss")] = each

    def get_qss(self, qss_name: str):
        if qss_name not in self._qss_dict:
            loguru.logger.warning(
                f"{Path(os.path.join(self._qss_dir, qss_name))}.qss not exist"
            )
            return False

        with open(self._qss_dict[qss_name], "r", encoding="UTF-8") as file:
            return file.read()


qss_loader = QssLoader()
