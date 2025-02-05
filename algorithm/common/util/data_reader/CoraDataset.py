#!/usr/bin/env python
# coding=utf-8

from algorithm.common.util.data_reader.DatesetRead import DatasetReader


class CoraDataset(DatasetReader):
    def __init__(self):
        super().__init__()
        # Raw Data Path
        self.data_path = r"./data/cora/cora.cites"
        # truthtable path
        self.truthtable_path = r"./data/cora/cora.content"
        # number of community
        self.number_of_community = 7  # 总共7个领域
        self.dataset_name = "CoraDataset"

    def read_data(self):
        """
        读取cora.cites，构建论文引用关系。
        :return: list[[paper_id_1, paper_id_2]], 表示paper_id_1引用了paper_id_2。
        """
        edges = set()
        node_map = {}  # 节点重新编号映射
        current_id = 0

        with open(self.data_path, "r", encoding="utf-8") as file:
            for line in file:
                cited, citing = map(int, line.split())

                # 如果节点未映射过，分配新编号
                if cited not in node_map:
                    node_map[cited] = current_id
                    current_id += 1
                if citing not in node_map:
                    node_map[citing] = current_id
                    current_id += 1

                # 将每个边按 (min, max) 顺序排列，确保无向图唯一性
                node1, node2 = node_map[citing], node_map[cited]
                edge = (min(node1, node2), max(node1, node2))
                edges.add(edge)  # 将tuple加入集合去重

        # 将集合转换为列表，且每个元素为list
        return [list(edge) for edge in edges]

    def read_truthtable(self):
        """
        读取cora.content，获取论文的所属领域。
        :return: list[[paper_id, community_label]], paper_id属于community_label表示的领域。
        """
        labels = {
            "Case_Based": 0,
            "Genetic_Algorithms": 1,
            "Neural_Networks": 2,
            "Probabilistic_Methods": 3,
            "Reinforcement_Learning": 4,
            "Rule_Learning": 5,
            "Theory": 6,
        }

        content = []
        node_map = {}  # 使用同一个node_map与read_data一致
        current_id = 0

        with open(self.truthtable_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                paper_id = int(parts[0])

                # 更新映射表确保与read_data中的一致
                if paper_id not in node_map:
                    node_map[paper_id] = current_id
                    current_id += 1

                label = parts[-1]  # 最后一个元素为领域
                community_label = labels[label]
                content.append([node_map[paper_id], community_label])

        return content
