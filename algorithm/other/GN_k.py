#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: GN_k.py
@date: 2024/10/15 15:48
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""

import networkx as nx
from networkx.algorithms.community import girvan_newman

from algorithm.common.util.decorator import time_record


@time_record
def GN_algorithm_k(edge_list, k, max_iter=100):
    """
    实现 Girvan-Newman 社区发现算法。

    :param edge_list: 输入的图的边列表
    :param k: 最终的社区数量
    :param max_iter: 最大迭代次数
    :return: 返回最佳社区划分（每个社区为一个列表）和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # GN 算法社区生成器
    communities_generator = girvan_newman(G)

    # 初始化社区数目
    iteration = 0
    best_communities = []

    for communities in communities_generator:
        community_list = [list(community) for community in communities]

        # 判断当前连通片数目
        if len(community_list) >= k or iteration >= max_iter:
            best_communities = community_list
            break

        iteration += 1

    return G, best_communities
