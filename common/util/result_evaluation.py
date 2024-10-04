#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: result_evaluation.py
@date: 2024/10/4 23:48
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""

import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict


class CommunityDetectionMetrics:

    # Todo 异常
    def __init__(self, G, communities, truth_table):
        self.G = G
        self.communities = communities
        self.truth_table = sorted(truth_table, key=lambda x: x[0])
        # 对齐真实标签和结果标签
        self.truth_to_result = self._align_labels()

    def _align_labels(self):
        """
        将算法输出的社区标签与真实标签对齐
        """
        # 从 truth 中提取真实标签
        truth_labels = [node[1] for node in self.truth_table]  # 只提取每个节点的真实标签

        # 将 result 转换为标签形式
        result_labels = self.convert_communities_to_labels()

        # 创建标签映射
        truth_to_result = defaultdict(int)

        for true_label in set(truth_labels):
            max_count = 0
            best_match = None

            for result_label in set(result_labels):
                # 计算真实标签和结果标签之间的匹配度
                count = sum(1 for tl, rl in zip(truth_labels, result_labels) if tl == true_label and rl == result_label)

                if count > max_count:
                    max_count = count
                    best_match = result_label

            truth_to_result[true_label] = best_match

        return truth_to_result

    def convert_communities_to_labels(self):
        # 创建一个空的标签列表，长度为节点数量
        num_nodes = max(max(community) for community in self.communities)  # 获取最大的节点编号
        labels = [-1] * num_nodes  # 初始化所有标签为 -1（表示未分配）

        # 将社区划分转化为标签
        for community_index, community in enumerate(self.communities):
            for node in community:
                labels[node - 1] = community_index  # 假设节点编号从 1 开始

        return labels

    def normalized_mutual_information(self):
        """
        计算归一化互信息 (NMI)
        """
        # 生成用于 NMI 计算的扁平化标签
        result_flat = [self.truth_to_result[community_index] for community_index, community in enumerate(self.communities)
                             for node in community]

        truth_flat = [node[1] for node in self.truth_table]  # 只提取真实标签

        # 计算 NMI
        nmi_value = normalized_mutual_info_score(truth_flat, result_flat)

        return nmi_value

    def accuracy(self):
        """
        计算社区划分的准确率 (ACC)
        """

        # 生成用于准确率计算的扁平化标签
        result_flat = [self.truth_to_result[community_index] for community_index, community in enumerate(self.communities)
                             for node in community]
        truth_flat = [node[1] for node in self.truth_table]  # 提取真实社区标签

        # 计算准确率
        correct_predictions = sum(1 for r, t in zip(result_flat, truth_flat) if r == t)
        acc_value = correct_predictions / len(truth_flat) if truth_flat else 0

        return acc_value

    def modularity(self):
        """
        计算模块度
        """
        return nx.algorithms.community.modularity(self.G, self.communities)

    def purity(self):
        """
        计算社区划分的纯度 (Purity)
        """
        # 计算每个社区的真实标签分布
        total_count = sum(len(community) for community in self.communities)  # 所有社区的节点总数
        correct_count = 0  # 记录正确的节点数

        # 为每个社区分配真实标签
        for community in self.communities:
            # 获取社区中所有节点的真实标签
            community_labels = [node for node in community]  # 这里直接是节点 ID

            # 计算每个标签在真实标签中的出现频率
            label_count = {label: 0 for label in set(node[1] for node in self.truth_table)}  # 初始化标签计数

            for node in community_labels:
                # 查找真实标签
                for truth_node in self.truth_table:
                    if truth_node[0] == node:
                        label_count[truth_node[1]] += 1

            # 找到出现次数最多的标签
            max_label_count = max(label_count.values(), default=0)
            correct_count += max_label_count  # 累加正确的节点数

        # 计算纯度
        purity_value = correct_count / total_count if total_count > 0 else 0
        return purity_value

    def adjusted_rand_index(self):
        """
        计算 ARI (Adjusted Rand Index)
        """
        # 从 result 中提取节点 ID
        result_flat = [self.truth_to_result[community_index] for community_index, community in enumerate(self.communities)
                             for node in community]
        # 从 truth 中提取真实标签
        truth_flat = [label[1] for label in self.truth_table]  # 这里假设 label[1] 是真实的社区标签

        return adjusted_rand_score(truth_flat, result_flat)

    def f1_score(self):
        """
        计算 F1 Score
        """

        # 从 result 中提取节点 ID
        result_flat = [self.truth_to_result[community_index] for community_index, community in enumerate(self.communities)
                             for node in community]
        # 从 truth 中提取真实标签
        truth_flat = [label[1] for label in self.truth_table]  # 这里假设 label[1] 是真实的社区标签

        # 计算精度和召回率
        precision = self.precision(result_flat, truth_flat)
        recall = self.recall(result_flat, truth_flat)

        # 计算 F1 Score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def conductance(self):
        """
        计算导电率
        """
        total_conductance = 0
        for community in self.communities:
            # 获取社区节点集合
            community_nodes = set(community)
            internal_edges = self.G.subgraph(community_nodes).number_of_edges()
            boundary_edges = sum(
                1 for node in community_nodes for neighbor in self.G.neighbors(node) if neighbor not in community_nodes)

            if internal_edges + boundary_edges > 0:
                conductance_value = boundary_edges / (internal_edges + boundary_edges)
                total_conductance += conductance_value

        return total_conductance / len(self.communities) if self.communities else 0

    def evaluate(self):
        """
        根据算法输出结果和真实标签计算所有评估指标
        """
        nmi = self.normalized_mutual_information()
        acc = self.accuracy()
        mod = self.modularity()
        pur = self.purity()
        ari = self.adjusted_rand_index()
        f1 = self.f1_score()
        cond = self.conductance()

        return {
            "NMI": nmi,
            "Accuracy": acc,
            "Modularity": mod,
            "Purity": pur,
            "ARI": ari,
            "F1 Score": f1,
            "Conductance": cond
        }

    def precision(self, predictions, truths):
        """
        计算精度
        """
        if not predictions:
            return 0.0
        correct_predictions = sum(1 for p in predictions if p in truths)
        return correct_predictions / len(predictions)

    def recall(self, predictions, truths):
        """
        计算召回率
        """
        if not truths:
            return 0.0
        correct_predictions = sum(1 for p in predictions if p in truths)
        return correct_predictions / len(truths)
