#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: Rcut.py
@date: 2024/10/15 15:49
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""

from sklearn.cluster import KMeans
import networkx as nx
from sklearn.metrics import jaccard_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

from common.util.decorator import time_record


@time_record
def Rcut_algorithm(edge_list, k):
    """
    实现 Rcut 谱聚类算法。

    :param edge_list: 输入的图的边列表
    :param k: 聚类的社区数量
    :return: 返回最佳社区划分（每个社区为一个列表）和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 构建邻接矩阵和度矩阵
    W = nx.to_numpy_array(G)
    D = np.diag(np.sum(W, axis=1))

    # 拉普拉斯矩阵 L = D - W
    L = D - W

    # 计算拉普拉斯矩阵的特征向量
    eigvals, eigvecs = np.linalg.eigh(L)

    # 取前 k 个特征向量
    X = eigvecs[:, :k]

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)

    # 根据聚类结果生成社区列表
    nodes = list(G.nodes)
    communities = {}
    for idx, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(nodes[idx])

    community_list = list(communities.values())

    return G, community_list
