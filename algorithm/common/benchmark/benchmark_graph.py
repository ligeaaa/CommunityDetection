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
    G: nx.Graph, number_of_communities, min_community_size, exponent, seed=42
):
    """
    生成符合幂律分布的社区大小序列，确保社区大小总和等于点的数量，并满足以下条件：
    1. 社区大小总和等于图中的节点数。
    2. 最小的社区大小大于最小度数，最大的社区大小大于最大度数。

    参数:
        G (nx.Graph): 目标图
        number_of_communities (int): 期望的社区数量
        min_community_size (int): 社区的最小大小
        exponent (float): 幂律指数 y (通常 y > 2)
        seed (int): 随机种子，确保可复现

    返回:
        community_sizes (list): 生成的社区大小序列
    """
    if exponent <= 1:
        raise ValueError("幂律指数 exponent 必须大于 1，否则期望值会发散。")

    np.random.seed(seed)
    N = G.number_of_nodes()  # 获取总节点数
    min_degree = min(dict(G.degree()).values())  # 最小度数
    max_degree = max(dict(G.degree()).values())  # 最大度数

    # 确保社区数量不超过节点数
    number_of_communities = min(number_of_communities, N)

    while True:
        # 生成符合幂律分布的社区大小
        raw_sizes = (
            np.random.pareto(exponent - 1, number_of_communities) + 1
        ) * min_community_size
        community_sizes = np.round(raw_sizes).astype(int)  # 取整

        # 归一化，使得社区大小总和等于 N
        community_sizes = (community_sizes / np.sum(community_sizes)) * N
        community_sizes = np.round(community_sizes).astype(int)

        # 确保所有社区大小至少大于 min_community_size
        community_sizes = np.maximum(community_sizes, min_community_size + 1)

        # 调整总和以精确匹配 N（避免由于取整导致误差）
        diff = N - np.sum(community_sizes)
        while diff != 0:
            idx = np.random.randint(0, len(community_sizes))
            community_sizes[idx] += np.sign(diff)
            diff = N - np.sum(community_sizes)

        # 确保最小社区大于最小度数，最大社区大于最大度数
        if min(community_sizes) > min_degree and max(community_sizes) > max_degree:
            break

    return community_sizes.tolist()


def assign_nodes_to_communities(G, communities_number_sequence, seed=42):
    """
    严格按照 communities_number_sequence 进行节点的社区分配。

    参数:
        G (nx.Graph): 生成的网络
        communities_number_sequence (list): 预期的社区大小序列
        seed (int): 随机种子，保证可复现性

    返回:
        dict: 映射节点到其社区的字典
    """
    random.seed(seed)
    nodes = list(G.nodes())

    # 1. 确保节点随机化
    random.shuffle(nodes)

    # 2. 社区大小排序(保证小社区先填充)
    sorted_communities = sorted(
        enumerate(communities_number_sequence), key=lambda x: x[1]
    )

    community_assignments = {}
    community_list = {i: set() for i, _ in sorted_communities}

    node_index = 0  # 追踪当前节点索引

    # 3. 按照社区大小严格分配
    for community_id, community_size in sorted_communities:
        while len(community_list[community_id]) < community_size and node_index < len(
            nodes
        ):
            node = nodes[node_index]
            community_list[community_id].add(node)
            community_assignments[node] = community_id
            node_index += 1

    return community_assignments


def create_graph(
    number_of_point,
    number_of_communities,
    min_community_size,
    degree_exponent,
    community_size_exponent,
    average_degree,
    min_degree,
    mixing_parameter,
    seed=42,
):
    random.seed(seed)
    # 生成degree序列
    degree_sequence = generate_power_law_degree_sequence(
        number_of_point, average_degree, min_degree, degree_exponent, seed=seed
    )
    # 使用配置模型生成图
    G = nx.configuration_model(degree_sequence, seed=seed)

    # 社区大小序列
    communities_number_sequence = generate_power_law_community_sequence(
        G, number_of_communities, min_community_size, community_size_exponent, seed=seed
    )

    # Assign nodes to communities
    node_communities = assign_nodes_to_communities(
        G, communities_number_sequence, seed=seed
    )
    # 将字典转换为列表，每个社区是一个节点 ID 列表
    communities = [[] for _ in range(len(communities_number_sequence))]
    for node, community_id in node_communities.items():
        communities[community_id].append(node)

    G = nx.Graph()
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

    return G, communities


if __name__ == "__main__":
    # 参数设定
    number_of_point = 200  # 节点数
    number_of_communities = 5
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 1.5  # 社区大小幂律指数
    average_degree = 5
    min_degree = 1
    min_community_size = 20
    mixing_parameter = 0.05  # 混合参数
    seed = random_seed

    # 生成图
    G, communities = create_graph(
        number_of_point,
        number_of_communities,
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
