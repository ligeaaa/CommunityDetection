#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: dataset_dealer.py
@date: 2024/10/3 14:42
Class Description:
- The data read util
@license: MIT
"""

from common.util.data_reader.DatesetRead import DatasetReader


class Dataset:
    def __init__(self, dataset_reader: DatasetReader):
        self.dataset_reader = dataset_reader()

    def read(self):
        return (self.dataset_reader.read_data(),
                self.dataset_reader.read_truthtable(),
                self.dataset_reader.number_of_community)
