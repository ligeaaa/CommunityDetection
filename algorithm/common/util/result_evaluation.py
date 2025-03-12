#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


class CommunityDetectionMetrics:

    def __init__(self, G, communities, truth_table):
        """
        :param G: nx.Graph()，输入的图
        :param communities:
            利用算法后划分的最佳社区，格式类似于：list:[[1, 2, 3, 4, 5], [6, 7]]，表示1-5节点属于0社区，6-7节点属于1社区
        :param truth_table:
            数据集的真实社区划分表，格式类似于：list:[[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]， 表明7和6节点属于0社区，5到1属于1社区
        """
        self.G = G
        self.communities = communities
        self.truth_table = truth_table
        # 将 truth_table 转化为 {node: label} 格式
        self.node_to_truth_label = {node: label for node, label in truth_table}

    def map_communities(self):
        """
        找到 communities 和 truth_table 的最优映射，通过计算标签之间的最佳匹配来进行映射
        使用匈牙利算法来最小化标签之间的差异，处理预测社区数量多于或少于真实社区的情况
        """
        # 获取真实标签
        truth_labels = [
            self.node_to_truth_label[node]
            for community in self.communities
            for node in community
        ]
        predicted_labels = []
        for i, community in enumerate(self.communities):
            predicted_labels.extend([i] * len(community))  # 将社区编号作为预测标签

        # 获取真实社区和预测社区的数量
        max_truth_label = max(truth_labels) + 1  # 真实标签的最大值+1
        num_predicted_labels = len(self.communities)  # 预测社区的数量

        # 生成混淆矩阵，行表示真实标签，列表示预测标签
        confusion_matrix = np.zeros((max_truth_label, num_predicted_labels))

        for t_label, p_label in zip(truth_labels, predicted_labels):
            confusion_matrix[t_label, p_label] += 1

        # 使用匈牙利算法找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # 创建映射字典 {预测标签: 真实标签}
        label_mapping = {}

        # 处理匹配到的社区
        for row, col in zip(row_ind, col_ind):
            label_mapping[col] = row

        # 处理未匹配的预测社区（如果有多余的预测社区）
        for col in range(num_predicted_labels):
            if col not in label_mapping:
                label_mapping[col] = -1  # 将多余的预测社区映射到 -1，表示无匹配

        # 处理未匹配的真实社区（如果有多余的真实社区）
        unmatched_truth_communities = set(range(max_truth_label)) - set(
            label_mapping.values()
        )
        for unmatched_truth in unmatched_truth_communities:
            # 将这些未匹配的真实社区映射到虚拟预测社区 -1
            label_mapping[len(label_mapping)] = unmatched_truth

        return label_mapping

    def apply_mapping(self, label_mapping):
        """
        使用找到的映射，将 communities 的预测标签映射为真实标签
        """
        mapped_labels = []
        for i, community in enumerate(self.communities):
            for node in community:
                mapped_labels.append(label_mapping[i])
        return mapped_labels

    def normalized_mutual_information(self):
        """
        计算归一化互信息 (NMI)。
        衡量两个不同划分之间相似性的指标，通常用于社区发现结果与真实社区划分之间的一致性评估。NMI 值在 0 到 1 之间，1 表示完全匹配，0 表示没有相关性。
        """
        truth_labels = [
            self.node_to_truth_label[node]
            for community in self.communities
            for node in community
        ]
        predicted_labels = []
        for i, community in enumerate(self.communities):
            predicted_labels.extend([i] * len(community))

        # 映射社区编号
        label_mapping = self.map_communities()
        mapped_labels = self.apply_mapping(label_mapping)

        # 计算 NMI
        nmi_value = normalized_mutual_info_score(truth_labels, mapped_labels)
        return nmi_value

    def accuracy(self):
        """
        计算社区划分的准确率 (ACC)。
        衡量社区划分中节点分类正确的比例。通过比较算法输出的社区标签与真实标签的匹配情况来计算准确率。
        """
        truth_labels = [
            self.node_to_truth_label[node]
            for community in self.communities
            for node in community
        ]
        predicted_labels = []
        for i, community in enumerate(self.communities):
            predicted_labels.extend([i] * len(community))

        # 映射社区编号
        label_mapping = self.map_communities()
        mapped_labels = self.apply_mapping(label_mapping)

        acc_value = accuracy_score(truth_labels, mapped_labels)
        return acc_value

    def modularity(self):
        """
        计算模块度 (Modularity)。
        表示社区划分的模块度得分，范围通常为 [-1, 1]，值越大表示划分越好。
        模块度是社区划分的经典评估指标之一，广泛用于衡量无监督社区发现算法的效果。
        """
        modularity_value = nx.algorithms.community.modularity(self.G, self.communities)
        return modularity_value

    def accuracy_per_community(self):
        """
        计算每个真实社区的 TP（真正例）、FP（假正例）、TN（真负例）、FN（假负例）。

        Returns:
            dict:
                {
                    "community_stats": {
                        真实社区: {
                            "TP": int,  # 该社区内正确匹配的点数
                            "FP": int,  # 该社区内被误分类到该社区的点数
                            "TN": int,  # 其他社区正确分类的点数
                            "FN": int   # 该社区内被误分类到其他社区的点数
                        }
                    }
                }
        """
        label_mapping = self.map_communities()
        community_stats = {}  # 存储每个真实社区的 TP, FP, TN, FN
        total_nodes = sum(len(community) for community in self.communities)  # 总节点数

        # 初始化每个真实社区的 TP, FP, TN, FN 统计值
        unique_truth_labels = set(self.node_to_truth_label.values())
        for label in unique_truth_labels:
            community_stats[label] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        # 计算 TP（真正例）和 FP（假正例）
        for i, community in enumerate(self.communities):
            mapped_label = label_mapping.get(
                i, -1
            )  # 获取匹配到的真实标签，若无匹配则为 -1
            if mapped_label == -1:
                continue  # 跳过未匹配的社区

            true_labels = [self.node_to_truth_label.get(node, -1) for node in community]
            correct_count = sum(1 for label in true_labels if label == mapped_label)
            wrong_count = len(community) - correct_count  # 误分类的点数

            # 统计 TP（真正例）和 FP（假正例）
            community_stats[mapped_label]["TP"] += correct_count
            community_stats[mapped_label]["FP"] += wrong_count

        # 计算 FN（假负例）和 TN（真负例）
        for label in unique_truth_labels:
            total_in_truth = sum(
                1
                for node, true_label in self.node_to_truth_label.items()
                if true_label == label
            )
            TP = community_stats[label]["TP"]
            FP = community_stats[label]["FP"]
            FN = total_in_truth - TP  # 假负例 = 真实属于该社区但被分类到其他社区的点
            TN = total_nodes - (TP + FP + FN)  # 真负例 = 其他社区正确分类的点

            community_stats[label]["FN"] = FN
            community_stats[label]["TN"] = TN

        return {"community_stats": community_stats}

    def evaluate(self):
        """
        根据算法输出结果和真实标签计算所有评估指标，并返回一个字典
        """
        nmi = self.normalized_mutual_information()
        acc = self.accuracy()
        mod = self.modularity()
        acc_per_community = self.accuracy_per_community()

        return {
            "NMI": nmi,
            "Accuracy": acc,
            "Modularity": mod,
            "acc_per_community": acc_per_community,
        }
