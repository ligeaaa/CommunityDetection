#!/usr/bin/env python
# coding=utf-8
import random
from itertools import combinations

import numpy as np
from networkx import Graph
from networkx.generators.community import stochastic_block_model

from algorithm.algorithm_dealer import Algorithm


class SBM(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "SBM"

    def process(self, G: Graph, **kwargs):
        # 使用初始分区、贪婪算法和蒙特卡罗方法生成 SBM 参数
        sizes, probs = self.estimate_sbm_parameters(
            G, num_clusters=kwargs["num_clusters"]
        )

        # 生成 SBM 随机块模型
        G_sbm = stochastic_block_model(sizes, probs)

        # 获取 SBM 模型生成的社区划分
        partition = G_sbm.graph["partition"]
        communities = [
            sorted(list(community)) for community in partition
        ]  # 转换为所需的格式

        return communities

    @staticmethod
    def estimate_sbm_parameters(G, num_clusters=2):
        """
        使用初始分区、贪婪算法和蒙特卡罗方法来估计 SBM 所需的参数。

        :param G: 输入图（nx.Graph）
        :param num_clusters: 社区数量
        :return: sizes, probs, 适用于 SBM 的社区大小和连接概率矩阵
        """
        # 随机初始分区，将节点分配到不同社区
        nodes = list(G.nodes())
        random.shuffle(nodes)
        community_size = len(nodes) // num_clusters
        initial_partition = [
            nodes[i * community_size : (i + 1) * community_size]
            for i in range(num_clusters)
        ]

        # 如果节点数不能被整除，剩余的节点分配到最后一个社区
        if len(nodes) % num_clusters != 0:
            initial_partition[-1].extend(nodes[num_clusters * community_size :])

        # 计算每个社区的大小
        sizes = [len(community) for community in initial_partition]

        # 使用蒙特卡罗方法计算社区间的连接概率
        probs = np.zeros((num_clusters, num_clusters))

        for i in range(num_clusters):
            for j in range(i, num_clusters):
                if i == j:
                    # 计算社区内部连接概率
                    community = initial_partition[i]
                    possible_edges = len(
                        list(combinations(community, 2))
                    )  # 内部所有可能的边数
                    actual_edges = sum(
                        1 for u, v in combinations(community, 2) if G.has_edge(u, v)
                    )
                    probs[i][j] = (
                        actual_edges / possible_edges if possible_edges > 0 else 0
                    )
                else:
                    # 计算社区之间的连接概率
                    community_i, community_j = (
                        initial_partition[i],
                        initial_partition[j],
                    )
                    possible_edges = len(community_i) * len(community_j)
                    actual_edges = sum(
                        1 for u in community_i for v in community_j if G.has_edge(u, v)
                    )
                    probs[i][j] = probs[j][i] = (
                        actual_edges / possible_edges if possible_edges > 0 else 0
                    )

        return sizes, probs
