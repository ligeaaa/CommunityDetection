#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: GN.py
@date: 2024/10/4 22:52
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
import random

import networkx as nx
from networkx.algorithms.community import girvan_newman

from common.util.decorator import time_record
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics


@time_record
def GN_algorithm(edge_list):
    # 创建一个无向图
    G = nx.Graph()

    # 将输入的 list 添加为图的边
    G.add_edges_from(edge_list)

    # GN 算法通过社区迭代生成器来实现
    communities_generator = girvan_newman(G)

    # 获取迭代器中的第一个划分结果
    try:
        first_community = next(communities_generator)
        communities = [list(community) for community in first_community]
    except StopIteration:
        communities = []  # 如果无法划分社区，返回空列表

    return G, communities


if __name__ == '__main__':
    # 示例输入：边的列表
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3], [2, 4], [4, 6], [6, 7]]
    truth_table = [[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]

    # edge_list = [[i, i + 1] for i in range(1, 301)] + [[random.randint(1, 300), random.randint(1, 300)] for _ in
    #                                                    range(100)]

    # 调用 GN 算法并返回图和社区结果
    G, communities = GN_algorithm(edge_list)

    # 原始图
    pos = nx.spring_layout(G)
    draw_communities(G, pos)

    # 返回结果，包括运行时间，正确率，可视化网络等
    draw_communities(G, pos, communities, draw_networkx_labels=True)

    evaluation = CommunityDetectionMetrics(G, communities, truth_table)

    metrics = evaluation.evaluate()

    # 打印评估结果
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
