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
        计算归一化互信息 (NMI)。
        衡量两个不同划分之间相似性的指标，通常用于社区发现结果与真实社区划分之间的一致性评估。NMI 值在 0 到 1 之间，1 表示完全匹配，0 表示没有相关性。
        """
        # 生成用于 NMI 计算的扁平化标签
        result_flat = []

        # 确保所有节点都有标签，并且不重复
        for community_index, community in enumerate(self.communities):
            for node in community:
                if node - 1 < len(self.truth_table):  # 确保节点编号在有效范围内
                    result_flat.append(community_index)

        truth_flat = [node[1] for node in self.truth_table]  # 真实标签

        # 确保 result_flat 和 truth_flat 的长度一致
        if len(truth_flat) != len(result_flat):
            raise ValueError(
                f"Inconsistent number of samples: truth_table has {len(truth_flat)} labels, but result has {len(result_flat)} labels.")

        # 计算 NMI
        nmi_value = normalized_mutual_info_score(truth_flat, result_flat)

        return nmi_value

    def accuracy(self):
        """
        计算社区划分的准确率 (ACC)。
        衡量社区划分中节点分类正确的比例。通过比较算法输出的社区标签与真实标签的匹配情况来计算准确率。
        反映算法生成的划分与真实标签的重合程度。
        """

        # 生成用于准确率计算的扁平化标签
        result_flat = [self.truth_to_result[community_index] for community_index, community in
                       enumerate(self.communities)
                       for node in community]
        truth_flat = [node[1] for node in self.truth_table]  # 提取真实社区标签

        # 计算准确率
        correct_predictions = sum(1 for r, t in zip(result_flat, truth_flat) if r == t)
        acc_value = correct_predictions / len(truth_flat) if truth_flat else 0

        return acc_value

    def modularity(self):
        """
        计算模块度 (Modularity)。
        表示社区划分的模块度得分，范围通常为 [-1, 1]，值越大表示划分越好。
        模块度是社区划分的经典评估指标之一，广泛用于衡量无监督社区发现算法的效果。
        """
        return nx.algorithms.community.modularity(self.G, self.communities)

    def evaluate(self):
        """
        根据算法输出结果和真实标签计算所有评估指标
        """
        nmi = self.normalized_mutual_information()
        acc = self.accuracy()
        mod = self.modularity()

        return {
            "NMI": nmi,
            "Accuracy": acc,
            "Modularity": mod
        }

