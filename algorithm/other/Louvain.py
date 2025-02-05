#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: Louvain.py
@date: 2024/10/14 15:30
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""

import community as community_louvain
import networkx as nx

from algorithm.common.util.decorator import time_record


@time_record
def Louvain_algorithm(edge_list, max_iter=10, modularity_threshold=0.5, frequency=2):
    """
    优化后的 Louvain 社区划分算法。

    :param edge_list: 输入的图的边列表
    :param max_iter: 最大迭代次数
    :param modularity_threshold: 模块度阈值，达到此值时提前停止
    :param frequency: 模块度计算的频率（每隔多少次迭代计算一次）
    :return: 返回最佳社区划分（每个社区为一个列表）和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 初始设置
    best_partition = None
    best_modularity = -1
    iteration = 0

    for _ in range(max_iter):
        # Louvain算法的社区划分
        partition = community_louvain.best_partition(G)

        # 将社区划分转换为列表形式
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        community_list = list(communities.values())

        # 每隔 'frequency' 次计算一次模块度
        if iteration % frequency == 0:
            modularity = community_louvain.modularity(partition, G)

            # 更新最佳社区划分
            if modularity > best_modularity:
                best_modularity = modularity
                best_partition = community_list  # 将最佳社区直接保存为列表形式

            # 当达到模块度阈值时提前停止
            if best_modularity >= modularity_threshold:
                print(f"提前停止，模块度达到阈值: {modularity_threshold}")
                break

        iteration += 1

    return G, best_partition
