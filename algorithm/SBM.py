#!/usr/bin/env python
# coding=utf-8
import networkx as nx
from community import community_louvain
from networkx.generators.community import stochastic_block_model
from common.util.decorator import time_record
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics
import numpy as np
from itertools import combinations  # 从 itertools 模块中导入 combinations


@time_record
def sbm_algorithm(edge_list, num_blocks=2):
    """
    自动生成 SBM 参数并应用 SBM 社区划分算法，输入与 GN_algorithm 格式一致。

    :param edge_list: 输入的图的边列表，格式为[[x, y], [x2, y2], ...]，表示在 x 和 y 之间有边
    :param num_blocks: 预期的社区数量（块数量）
    :return: 返回基于 SBM 模型生成的图及其社区划分
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)  # 从 edge_list 中添加边，构建图

    # 使用 Louvain 或其他算法初步划分社区（此处省略，可以根据需要修改）
    sizes, probs = estimate_sbm_parameters(G)

    # 生成 SBM 随机块模型
    G_sbm = stochastic_block_model(sizes, probs)

    # 获取 SBM 模型生成的社区划分
    partition = G_sbm.graph['partition']
    best_communities = [sorted(list(community)) for community in partition]

    return G_sbm, best_communities


def estimate_sbm_parameters(G):
    """
    根据给定的图 G 自动估计 SBM 所需的参数。

    :param G: 输入图（nx.Graph）
    :return: sizes, probs, 适用于 SBM 的社区大小和连接概率矩阵
    """
    # 使用 Louvain 算法进行初步社区划分，确定社区数量和节点分布
    partition = community_louvain.best_partition(G)

    # 根据社区划分结果生成社区大小
    communities_dict = {}
    for node, community in partition.items():
        if community not in communities_dict:
            communities_dict[community] = []
        communities_dict[community].append(node)

    # 动态获取实际的社区数量
    num_blocks = len(communities_dict)  # 实际社区的数量
    sizes = [len(nodes) for nodes in communities_dict.values()]

    # 计算每个社区的连接概率
    probs = np.zeros((num_blocks, num_blocks))  # 使用动态大小来匹配社区数量

    for i, community_i in enumerate(communities_dict.values()):
        for j, community_j in enumerate(communities_dict.values()):
            if i <= j:  # 计算上三角
                if i == j:
                    possible_edges = len(list(combinations(community_i, 2)))  # 使用 itertools.combinations
                    actual_edges = sum(1 for u, v in combinations(community_i, 2) if G.has_edge(u, v))
                else:
                    possible_edges = len(community_i) * len(community_j)
                    actual_edges = sum(1 for u in community_i for v in community_j if G.has_edge(u, v))

                if possible_edges > 0:
                    probs[i][j] = actual_edges / possible_edges
                    probs[j][i] = probs[i][j]

    return sizes, probs


if __name__ == '__main__':
    # 示例输入：边的列表
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3], [2, 4], [4, 6], [6, 7]]
    truth_table = [[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]  # 真实社区标签

    # 调用 SBM 算法并返回图和社区结果
    G_sbm, communities_sbm = sbm_algorithm(edge_list)

    # 可视化结果
    pos = nx.spring_layout(G_sbm)
    draw_communities(G_sbm, pos)

    # 返回结果，包括运行时间，正确率，可视化网络等
    draw_communities(G_sbm, pos, communities_sbm, draw_networkx_labels=True)

    evaluation = CommunityDetectionMetrics(G_sbm, communities_sbm, truth_table)

    metrics = evaluation.evaluate()

    # 打印评估结果
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
