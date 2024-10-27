#!/usr/bin/env python
# coding=utf-8
from common.util.data_reader.DatesetRead import DatasetReader


class ZKClubDataset(DatasetReader):
    def __init__(self):
        # Raw Data Path
        super().__init__()
        self.data_path = r".\\data\\ZK Club\\karate_club_data.txt"
        # truthtable path
        self.truthtable_path = r".\\data\\ZK Club\\truthtable.txt"
        # number of community
        self.number_of_community = 2
        # name of dataset
        self.dataset_name = "ZKClubDataset"

    def read_data(self):
        """
        read raw data
        :return: list[a, b], Person a and Person b have email correspondence.
        """
        content = []
        min_person = float('inf')  # 初始化最小值为无穷大

        # 读取文件并找到最小编号
        with open(self.data_path, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                # 将每行的内容转化为数字
                numbers = [int(num) for num in line.split()]
                content.append(numbers)

                # 找到最小的人员编号
                min_person = min(min_person, *numbers)

        # 如果最小编号不是 0，则将所有人员编号减去最小值
        if min_person != 0:
            content = [[num - min_person for num in pair] for pair in content]

        return content

    def read_truthtable(self):
        """
        read truthtable data
        :return: list[a, b], Person a belongs to Department b.
        """
        content = []
        min_person = float('inf')  # 初始化最小人员编号
        min_department = float('inf')  # 初始化最小部门编号

        # 读取文件并找到最小编号
        with open(self.truthtable_path, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                # 将每行的内容转化为数字
                numbers = [int(num) for num in line.split()]
                content.append(numbers)

                # 找到最小的人员编号和部门编号
                min_person = min(min_person, numbers[0])
                min_department = min(min_department, numbers[1])

        # 如果人员编号或部门编号不是从0开始，则调整它们
        if min_person != 0 or min_department != 0:
            content = [[person - min_person, department - min_department] for person, department in content]

        return content

    def get_base_information(self):
        ...