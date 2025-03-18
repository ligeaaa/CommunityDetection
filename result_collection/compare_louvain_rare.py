#!/usr/bin/env python
# coding=utf-8
import time

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer
from algorithm.classic.RareDetection import RareDetection
from algorithm.classic.louvain import Louvain
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics

if __name__ == "__main__":
    # 参数设定
    number_of_point = 200  # 节点数
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 3  # 社区大小幂律指数
    average_degree = 6
    min_degree = 2
    min_community_size = 15
    mixing_parameter = 0.1  # 混合参数
    # 参数设定
    # number_of_point = 300  # 节点数
    # degree_exponent = 3  # 幂律指数
    # community_size_exponent = 3  # 社区大小幂律指数
    # average_degree = 6
    # min_degree = 2
    # min_community_size = 15
    # mixing_parameter = 0.1  # 混合参数
    # 生成图
    G, true_communities = create_graph(
        number_of_point,
        min_community_size,
        degree_exponent,
        community_size_exponent,
        average_degree,
        min_degree,
        mixing_parameter,
        seed=53,
    )

    algorithmDealer = AlgorithmDealer()
    rare_algorithm = RareDetection()
    louvain = Louvain()
    results = algorithmDealer.run([rare_algorithm, louvain], G, num_clusters=6)
    truth_table = [
        [node, community_id]
        for community_id, nodes in enumerate(true_communities)
        for node in reversed(nodes)
    ]
    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos, true_communities)
    for result in results:
        time.sleep(1)
        communities = result.communities
        algorithm_name = result.algorithm_name

        # 计算评估指标
        evaluation = CommunityDetectionMetrics(G, communities, truth_table)
        metrics = evaluation.evaluate()
        metrics["runtime"] = result.runtime

        # 可视化结果
        draw_communities(G, pos, communities, title=algorithm_name, metrics=metrics)

    print(1)
