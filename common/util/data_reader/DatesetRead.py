#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: DatesetRead.py
@date: 2024/10/3 15:05
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""


class DatasetReader:
    def __init__(self):
        # Raw Data Path
        self.data_path = ...
        # truthtable path
        self.truthtable_path = ...
        # number of community
        self.number_of_community = ...

    def read_data(self):
        '''
        read raw data
        :return: list
        '''
        ...

    def read_truthtable(self):
        '''
        read truthtable data
        :return: list
        '''
        ...

    def get_base_information(self):
        ...