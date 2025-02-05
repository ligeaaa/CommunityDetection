#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: AlinkJaccard.py
@date: 2024/10/15 15:49
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import jaccard_score

from algorithm.common.util.decorator import time_record


@time_record
def Alink_Jaccard_algorithm(edge_list, k):
    """
    实现 Alink-Jaccard 层次聚类算法。

    :param edge_list: 输入的图的边列表
    :param k: 最终的社区数量
    :return: 返回最佳社区划分（每个社区为一个列表）和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 计算 Jaccard 相似度矩阵
    nodes = list(G.nodes)
    adj_matrix = nx.to_numpy_array(G)
    jaccard_matrix = np.zeros((len(nodes), len(nodes)))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            jaccard_matrix[i][j] = jaccard_score(adj_matrix[i], adj_matrix[j])
            jaccard_matrix[j][i] = jaccard_matrix[i][j]

    # 进行层次聚类
    Z = linkage(jaccard_matrix, method="average")
    labels = fcluster(Z, k, criterion="maxclust")

    # 将聚类结果转换为社区列表
    communities = {}
    for idx, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(nodes[idx])

    community_list = list(communities.values())

    return G, community_list
