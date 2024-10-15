#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: Modularity.py
@date: 2024/10/15 15:49
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

from common.util.decorator import time_record


@time_record
def Modularity_algorithm(edge_list, k):
    """
    实现基于模块度的谱聚类算法。

    :param edge_list: 输入的图的边列表
    :param k: 聚类的社区数量
    :return: 返回最佳社区划分（每个社区为一个列表）和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 构建邻接矩阵 A 和度向量 d
    A = nx.to_numpy_array(G)
    d = np.sum(A, axis=1)
    m = np.sum(A) / 2

    # 构建 B 矩阵 B = A - dd^T / (2m)
    B = A - np.outer(d, d) / (2 * m)

    # 计算 B 矩阵的特征向量
    eigvals, eigvecs = np.linalg.eigh(B)

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
