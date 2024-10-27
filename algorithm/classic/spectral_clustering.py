#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from common.util.decorator import time_record
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics

# @time_record
def spectral_clustering_algorithm(edge_list, num_clusters=2):
    """
    谱聚类社区划分算法，支持从0或1开始的节点编号。

    :param edge_list: 输入的图的边列表
    :param num_clusters: 预期的社区数量
    :return: 返回最佳社区划分和对应的图
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # 获取邻接矩阵
    adj_matrix = nx.to_numpy_array(G)

    # 使用谱聚类算法进行社区划分
    spectral_model = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral_model.fit_predict(adj_matrix)

    # 检测节点编号是否从 0 开始或 1 开始
    min_node = min(G.nodes())

    # 根据标签进行社区划分，调整节点编号
    communities = {}
    for node, label in zip(G.nodes(), labels):  # 使用 G.nodes() 中的实际节点编号
        if label not in communities:
            communities[label] = []
        communities[label].append(node)

    # 确保社区划分没有遗漏节点
    all_nodes = set(G.nodes())  # 图中所有节点
    assigned_nodes = set(node for community in communities.values() for node in community)  # 所有已分配的节点
    unassigned_nodes = all_nodes - assigned_nodes  # 没有被分配的节点

    # 仅当确实有未分配节点时，将它们归为一个单独的社区
    if unassigned_nodes:
        communities[len(communities)] = list(unassigned_nodes)

    best_communities = [sorted(nodes) for nodes in communities.values()]

    return best_communities


if __name__ == '__main__':
    # 示例输入：边的列表
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3], [2, 4], [4, 6], [6, 7]]
    truth_table = [[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]  # 真实社区标签

    # 调用谱聚类算法并返回图和社区结果
    G, best_communities = spectral_clustering_algorithm(edge_list, num_clusters=2)

    # 可视化结果
    pos = nx.spring_layout(G)
    draw_communities(G, pos)
    draw_communities(G, pos, best_communities, draw_networkx_labels=True)

    # 评估
    evaluation = CommunityDetectionMetrics(G, best_communities, truth_table)
    metrics = evaluation.evaluate()

    for metric, value in metrics.items():
        print(f"{metric}: {value}")
