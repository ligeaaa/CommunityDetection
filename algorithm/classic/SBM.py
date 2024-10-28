#!/usr/bin/env python
# coding=utf-8
import networkx as nx
from networkx.generators.community import stochastic_block_model
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics
import numpy as np
import random
from itertools import combinations


def sbm_algorithm(edge_list, num_blocks=2):
    """
    自动生成 SBM 参数并应用 SBM 社区划分算法，输入与 GN_algorithm 格式一致。

    :param edge_list: 输入的图的边列表，格式为[[x, y], [x2, y2], ...]，表示在 x 和 y 之间有边
    :param num_blocks: 预期的社区数量（块数量）
    :return: communities, list 格式，如 [[1, 2, 3, 4, 5], [6, 7]]，表示节点划分的社区
    """
    # 创建无向图
    G = nx.Graph()
    G.add_edges_from(edge_list)  # 从 edge_list 中添加边，构建图

    # 使用初始分区、贪婪算法和蒙特卡罗方法生成 SBM 参数
    sizes, probs = estimate_sbm_parameters(G, num_blocks)

    # 生成 SBM 随机块模型
    G_sbm = stochastic_block_model(sizes, probs)

    # 获取 SBM 模型生成的社区划分
    partition = G_sbm.graph['partition']
    communities = [sorted(list(community)) for community in partition]  # 转换为所需的格式

    return communities


def estimate_sbm_parameters(G, num_blocks):
    """
    使用初始分区、贪婪算法和蒙特卡罗方法来估计 SBM 所需的参数。

    :param G: 输入图（nx.Graph）
    :param num_blocks: 社区数量
    :return: sizes, probs, 适用于 SBM 的社区大小和连接概率矩阵
    """
    # 随机初始分区，将节点分配到不同社区
    nodes = list(G.nodes())
    random.shuffle(nodes)
    community_size = len(nodes) // num_blocks
    initial_partition = [nodes[i * community_size:(i + 1) * community_size] for i in range(num_blocks)]

    # 如果节点数不能被整除，剩余的节点分配到最后一个社区
    if len(nodes) % num_blocks != 0:
        initial_partition[-1].extend(nodes[num_blocks * community_size:])

    # 计算每个社区的大小
    sizes = [len(community) for community in initial_partition]

    # 使用蒙特卡罗方法计算社区间的连接概率
    probs = np.zeros((num_blocks, num_blocks))

    for i in range(num_blocks):
        for j in range(i, num_blocks):
            if i == j:
                # 计算社区内部连接概率
                community = initial_partition[i]
                possible_edges = len(list(combinations(community, 2)))  # 内部所有可能的边数
                actual_edges = sum(1 for u, v in combinations(community, 2) if G.has_edge(u, v))
                probs[i][j] = actual_edges / possible_edges if possible_edges > 0 else 0
            else:
                # 计算社区之间的连接概率
                community_i, community_j = initial_partition[i], initial_partition[j]
                possible_edges = len(community_i) * len(community_j)
                actual_edges = sum(1 for u in community_i for v in community_j if G.has_edge(u, v))
                probs[i][j] = probs[j][i] = actual_edges / possible_edges if possible_edges > 0 else 0

    return sizes, probs


if __name__ == '__main__':
    # 示例输入：边的列表
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3], [2, 4], [4, 6], [6, 7]]
    truth_table = [[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]  # 真实社区标签

    # 调用 SBM 算法并返回社区结果
    communities_sbm = sbm_algorithm(edge_list, num_blocks=2)
    print("社区划分结果:", communities_sbm)

    # 可视化结果
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G)
    draw_communities(G, pos, communities_sbm, draw_networkx_labels=True)

    # 返回结果，包括运行时间，正确率，可视化网络等
    evaluation = CommunityDetectionMetrics(G, communities_sbm, truth_table)
    metrics = evaluation.evaluate()

    # 打印评估结果
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
