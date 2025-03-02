#!/usr/bin/env python
# coding=utf-8

import numpy as np
import networkx as nx
import random

from algorithm.common.constant.constant_number import random_seed
from algorithm.common.util.drawer import draw_communities


def generate_power_law_degree_sequence(N, avg_degree, min_degree, exponent, seed=42):
    """
    生成一个符合幂律分布的度数序列，满足指定的平均度数。

    参数:
        N (int): 节点数
        avg_degree (float): 期望的平均度数
        exponent (float): 幂律指数 y (通常 y > 2)

    返回:
        degree_sequence (list): 生成的度数序列
    """
    if exponent <= 1:
        raise ValueError("幂律指数 exponent 必须大于 1，否则期望值会发散。")

    # 生成符合幂律分布的随机度数
    np.random.seed(seed)
    raw_degrees = (np.random.pareto(exponent - 1, N) + 1) * min_degree

    # 归一化以匹配期望的平均度数
    raw_degrees *= avg_degree / np.mean(raw_degrees)

    # 取整数，并确保最小度数不小于 1
    degree_sequence = np.round(raw_degrees).astype(int)
    degree_sequence = np.maximum(degree_sequence, 1)  # 防止出现度数为 0 的情况

    # 确保度数序列的总和为偶数（否则调整最大度数的节点）
    if np.sum(degree_sequence) % 2 == 1:
        degree_sequence[np.argmax(degree_sequence)] += 1

    return degree_sequence.tolist()


def generate_power_law_community_sequence(
    number_of_point,
    degree_sequence,
    exponent,
    min_community_size,
    max_community_size=None,
    seed=42,
):
    """
    生成符合幂律分布的社区大小序列，确保社区大小总和等于点的数量，并满足以下条件：
    1. 社区大小总和等于图中的节点数。
    2. 最小的社区大小大于最小度数，最大的社区大小大于最大度数。

    参数:
        G (nx.Graph): 目标图
        min_community_size (int): 社区的最小大小
        exponent (float): 幂律指数 y (通常 y > 2)
        seed (int): 随机种子，确保可复现

    返回:
        community_sizes (list): 生成的社区大小序列
    """
    if exponent <= 1:
        raise ValueError("幂律指数 exponent 必须大于 1，否则期望值会发散。")
    if max_community_size is None:
        max_community_size = number_of_point

    np.random.seed(seed)
    random.seed(seed)
    min_degree = min(degree_sequence)  # 最小度数
    min_community_size = max(min_community_size, min_degree + 1)
    # 生成符合幂律分布的社区大小
    powerlaw_sequence = (
        np.random.pareto(exponent - 1, number_of_point) + 1
    ) * min_community_size
    powerlaw_sequence = np.round(powerlaw_sequence)  # 取整
    powerlaw_sequence = powerlaw_sequence[
        powerlaw_sequence <= max_community_size
    ].astype(int)

    flag = False
    while not flag:
        # 初始化社区size
        community_sizes = []
        total_community_size = 0
        # 生成初始的社区size序列
        while total_community_size < number_of_point:
            a = random.choice(powerlaw_sequence)
            if total_community_size + a > number_of_point:
                break

            community_sizes.append(a)
            total_community_size += a

        # 调整最后一个社区大小，确保总和等于 `number_of_point`
        if total_community_size < number_of_point:
            remaining_size = number_of_point - total_community_size
            if remaining_size >= min_community_size:
                community_sizes.append(remaining_size)
                flag = True

    return community_sizes


def assign_nodes_to_communities(
    N, mixing_parameter, degree_sequence, communities_number_sequence, seed=42
):
    """
    严格按照 communities_number_sequence 进行节点的社区分配，确保社区不会超出预期大小。

    参数:
        N (int): 图中的节点总数
        mixing_parameter (float): 混合参数
        degree_sequence (list): 每个节点的度数序列
        communities_number_sequence (list): 预期的社区大小序列
        seed (int): 随机种子，保证可复现性

    返回:
        dict: 映射节点到其社区的字典
    """
    random.seed(seed)

    # 打乱节点顺序
    nodes = list(range(N))
    random.shuffle(nodes)

    # 记录每个社区当前的分配数量
    community_counts = {i: 0 for i in range(len(communities_number_sequence))}

    # 存储分配结果
    community_assignments = {}

    for node in nodes:
        # TODO 添加一个变量存储可选择的社区，降低复杂度
        while True:
            # 选择一个随机社区
            community_id = random.choice(range(len(communities_number_sequence)))

            # 检查社区是否已满
            if (
                community_counts[community_id]
                < communities_number_sequence[community_id]
            ):
                # 确保该社区的大小满足条件
                if (1 - mixing_parameter) * degree_sequence[
                    node
                ] < communities_number_sequence[community_id]:
                    # 分配节点到该社区
                    community_assignments[node] = community_id
                    community_counts[community_id] += 1  # 更新社区成员数
                    break  # 退出循环，分配完成

    return community_assignments


def create_graph(
    number_of_point,
    min_community_size,
    degree_exponent,
    community_size_exponent,
    average_degree,
    min_degree,
    mixing_parameter,
    whether_simple_graph=True,
    seed=42,
):
    """

    Args:
        number_of_point:
        min_community_size:
        degree_exponent:
        community_size_exponent:
        average_degree:
        min_degree:
        mixing_parameter:
        whether_simple_graph:
        seed:

    Returns:

    References:
    [1] Lancichinetti, A., Fortunato, S. and Radicchi, F. (2008) ‘Benchmark graphs for testing community detection
    algorithms’, Physical Review E, 78(4), p. 046110. Available at: https://doi.org/10.1103/PhysRevE.78.046110.
    [2] https://www.osgeo.cn/networkx/reference/generated/networkx.generators.community.LFR_benchmark_graph.html

    """
    random.seed(seed)
    # 生成degree序列
    degree_sequence = generate_power_law_degree_sequence(
        number_of_point, average_degree, min_degree, degree_exponent, seed=seed
    )

    # 社区大小序列
    communities_number_sequence = generate_power_law_community_sequence(
        number_of_point,
        degree_sequence,
        community_size_exponent,
        min_community_size,
        seed=seed,
    )

    # Assign nodes to communities
    node_communities = assign_nodes_to_communities(
        number_of_point,
        mixing_parameter,
        degree_sequence,
        communities_number_sequence,
        seed=seed,
    )
    # 将字典转换为列表，每个社区是一个节点 ID 列表
    communities = [[] for _ in range(len(communities_number_sequence))]
    for node, community_id in node_communities.items():
        communities[community_id].append(node)

    # 根据mixing_parameter排线
    G = nx.MultiGraph()
    G.add_nodes_from(range(number_of_point))
    for c in communities:
        for u in c:
            while G.degree(u) < round(degree_sequence[u] * (1 - mixing_parameter)):
                v = random.choice(list(c))
                G.add_edge(u, v)
            while G.degree(u) < degree_sequence[u]:
                v = random.choice(range(number_of_point))
                if v not in c:
                    G.add_edge(u, v)
            G.nodes[u]["community"] = c

    if whether_simple_graph:
        # 去除重边和自连
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

    return G, communities


# TODO 写注释
if __name__ == "__main__":
    # 参数设定
    number_of_point = 200  # 节点数
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 1.5  # 社区大小幂律指数
    average_degree = 5
    min_degree = 1
    min_community_size = 5
    mixing_parameter = 0.05  # 混合参数
    seed = random_seed

    # 生成图
    G, communities = create_graph(
        number_of_point,
        min_community_size,
        degree_exponent,
        community_size_exponent,
        average_degree,
        min_degree,
        mixing_parameter,
        seed,
    )

    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos)

    # 返回结果，包括运行时间，正确率，可视化网络等
    draw_communities(G, pos, communities)

    original_modularity = nx.algorithms.community.modularity(G, communities)
    print(original_modularity)
    print(communities)
