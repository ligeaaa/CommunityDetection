#!/usr/bin/env python
# coding=utf-8
from algorithm.common.util.data_reader.DatesetRead import DatasetReader


class DBLPDataset(DatasetReader):
    def __init__(self):
        # Raw Data Path
        super().__init__()
        self.data_path = r"./data/DBLP/com-dblp.ungraph.txt"
        # truthtable path
        self.truthtable_path = r"./data/amazon/com-amazon.top5000.cmty.txt"
        # Number of communities,可以在get_base_information或其他逻辑中确定实际的社区数量
        self.number_of_community = 13477
        # name of dataset
        self.dataset_name = "DBLPDataset"

    def read_data(self):
        """
        Reads the raw edge data.
        :return: list[[a, b]], where a and b are nodes with an undirected edge between them.
        """
        content = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 忽略注释和空行
                if line.startswith("#") or not line.strip():
                    continue

                # Split line into two integers representing an edge
                numbers = line.split()
                numbers = [int(num) for num in numbers]
                content.append(numbers)

        return content

    def read_truthtable(self):
        """
        Reads the community data for each node.
        :return: list[[x, y]], where x is a node and y is the community it belongs to.
        """
        content = []
        with open(self.truthtable_path, 'r', encoding='utf-8') as file:
            community_id = 0  # 初始社区编号
            for line in file:
                # Ignore empty lines
                if not line.strip():
                    continue

                # Split line into integers, each node in the community
                nodes = line.split()
                nodes = [int(node) for node in nodes]

                # Append each node with the current community ID
                content.extend([[node, community_id] for node in nodes])
                community_id += 1

        # 更新社区数量
        self.number_of_community = community_id
        return content

    def get_base_information(self):
        """
        Provides basic information about the dataset, including the number of communities.
        """
        print(f"DBLP Dataset - Total Nodes: {len(set(node for edge in self.read_data() for node in edge))}")
        print(f"Total Communities: {self.number_of_community}")
