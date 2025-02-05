#!/usr/bin/env python
# coding=utf-8


class DatasetReader:
    def __init__(self):
        # Raw Data Path
        self.data_path = ...
        # truthtable path
        self.truthtable_path = ...
        # number of community
        self.number_of_community = ...
        # name of dataset
        self.dataset_name = ...

    def read_data(self):
        """
        read raw data
        :return: list
        """
        ...

    def read_truthtable(self):
        """
        read truthtable data
        :return: list
        """
        ...

    def get_base_information(self): ...
