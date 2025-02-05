#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: SpectralClustering.py
@date: 2024/10/14 15:30
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
import networkx as nx
from community import community_louvain
from sklearn.cluster import SpectralClustering

from algorithm.common.util.decorator import time_record


@time_record
def SpectralClustering_algorithm(
    edge_list, n_clusters=2, max_iter=10, modularity_threshold=0.5, frequency=2
):
    """
    优化后的 谱聚类 社区划分算法。

    :param edge_list: 输入的图的边列表
    :param n_clusters: 聚类的社区数量
    :param max_iter: 最大迭代次数
    :param modularity_threshold: 模块度阈值，达到此值时提前停止
    :param frequency: 模块度计算的频率（每隔多少次迭代计算一次）
    :return: 返回最佳社区划分（每个社区为一个列表）和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 将图转化为邻接矩阵
    adj_matrix = nx.to_numpy_array(G)

    best_modularity = -1
    iteration = 0
    best_communities = []

    for _ in range(max_iter):
        # 谱聚类算法
        spectral_model = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
        )
        labels = spectral_model.fit_predict(adj_matrix)

        # 将聚类结果转化为社区列表
        communities = {}
        for node, label in enumerate(labels):
            if label not in communities:
                communities[label] = []
            communities[label].append(node + 1)  # 节点编号从1开始

        community_list = list(communities.values())

        # 将社区列表转换为节点到社区的字典
        partition = {
            node: label
            for label, community in enumerate(community_list)
            for node in community
        }

        # 每隔 'frequency' 次计算一次模块度
        if iteration % frequency == 0:
            modularity = community_louvain.modularity(
                partition, G
            )  # 利用louvain模块的modularity计算

            # 更新最佳社区划分
            if modularity > best_modularity:
                best_modularity = modularity
                best_communities = community_list  # 将最佳社区直接保存为列表形式

            # 当达到模块度阈值时提前停止
            if best_modularity >= modularity_threshold:
                print(f"提前停止，模块度达到阈值: {modularity_threshold}")
                break

        iteration += 1

    return G, best_communities
