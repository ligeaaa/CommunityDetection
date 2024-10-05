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
import networkx.algorithms.community as community_louvain

from common.util.decorator import time_record
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics


@time_record
def GN_algorithm(edge_list, max_iter=10, modularity_threshold=0.5, frequency=2):
    """
    优化后的 Girvan-Newman 社区划分算法。

    :param edge_list: 输入的图的边列表
    :param max_iter: 最大迭代次数
    :param modularity_threshold: 模块度阈值，达到此值时提前停止
    :param frequency: 模块度计算的频率（每隔多少次迭代计算一次）
    :return: 返回最佳社区划分和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # GN 算法社区生成器
    communities_generator = girvan_newman(G)

    # 初始设置
    best_communities = None
    best_modularity = -1
    iteration = 0

    for communities in communities_generator:
        # 每隔 'frequency' 次计算一次模块度，减少频率
        if iteration % frequency == 0:
            community_list = [list(community) for community in communities]
            modularity = community_louvain.modularity(G, community_list)

            # 更新最佳社区划分
            if modularity > best_modularity:
                best_modularity = modularity
                best_communities = community_list

            # 当达到模块度阈值时提前停止
            if best_modularity >= modularity_threshold:
                print(f"提前停止，模块度达到阈值: {modularity_threshold}")
                break

        # 达到最大迭代次数时停止
        if iteration >= max_iter:
            print(f"达到最大迭代次数: {max_iter}")
            break

        iteration += 1

    return G, best_communities


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
