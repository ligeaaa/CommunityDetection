#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: EmailEuCoreDataset.py
@date: 2024/10/3 15:10
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
from common.util.data_reader.DatesetRead import DatasetReader


class EmailEuCoreDataset(DatasetReader):
    def __init__(self):
        # Raw Data Path
        super().__init__()
        self.data_path = ".\\data\\email-Eu-core network\\email-Eu-core.txt"
        # truthtable path
        self.truthtable_path = ".\\data\\email-Eu-core network\\email-Eu-core-department-labels.txt"
        # number of community
        self.number_of_community = 42

    def read_data(self):
        """
        read raw data
        :return: list[a, b], Person a and Person b have email correspondence.
        """
        content = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                # change the content into number
                numbers = line.split()
                numbers = [int(num) for num in numbers]
                content.append(numbers)

        return content

    def read_truthtable(self):
        """
        read truthtable data
        :return: list[a, b], Person a belongs to Department b.
        """
        content = []
        with open(self.truthtable_path, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                # change the content into number
                numbers = line.split()
                numbers = [int(num) for num in numbers]
                content.append(numbers)

        return content

    def get_base_information(self):
        ...