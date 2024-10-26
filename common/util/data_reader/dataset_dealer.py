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
        self.dataset_reader = dataset_reader

    def read(self):
        """
        Reads the dataset and returns network edge information, node community assignments, and the number of communities.

        :return:
            raw_data: list of [int, int]
                Each element is [x, y], indicating an edge exists between nodes x and y.
            truth_table: list of [int, int]
                Each element is [x, y], indicating node x belongs to community y.
            number_of_community: int
                The total number of communities.
        """
        raw_data = self.dataset_reader.read_data()
        truth_table = self.dataset_reader.read_truthtable()
        number_of_community = self.dataset_reader.number_of_community
        return raw_data, truth_table, number_of_community
