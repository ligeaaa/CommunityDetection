#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import community as community_louvain
from common.util.decorator import time_record
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics

# @time_record
def louvain_algorithm(edge_list):
    """
    Louvain社区划分算法。

    :param edge_list: 输入的图的边列表
    :return: 返回最佳社区划分和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 使用Louvain算法进行社区划分
    partition = community_louvain.best_partition(G)

    # 将节点根据其社区编号进行分组
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    best_communities = [sorted(nodes) for nodes in communities.values()]  # 按节点ID排序

    return G, best_communities

if __name__ == '__main__':
    # 示例输入：边的列表
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3], [2, 4], [4, 6], [6, 7]]
    truth_table = [[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]

    # 调用 Louvain 算法并返回图和社区结果
    G, best_communities = louvain_algorithm(edge_list)

    # 可视化结果
    pos = nx.spring_layout(G)
    draw_communities(G, pos)
    draw_communities(G, pos, best_communities, draw_networkx_labels=True)

    # 评估
    evaluation = CommunityDetectionMetrics(G, best_communities, truth_table)
    metrics = evaluation.evaluate()

    for metric, value in metrics.items():
        print(f"{metric}: {value}")