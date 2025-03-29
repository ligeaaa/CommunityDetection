#!/usr/bin/env python
# coding=utf-8
import time

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer
from algorithm.classic.Leiden_Rare import Leiden_Rare
from algorithm.classic.leiden import Leiden
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.CommunityCompare import CommunityComparator
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics

if __name__ == "__main__":
    # 参数设定
    # import random
    # random.seed(55)
    # number_of_point = int(random.random() * 300)  # 节点数
    # degree_exponent = 3  # 幂律指数
    # community_size_exponent = random.random()*2+1  # 社区大小幂律指数
    # average_degree = int(random.random() * 5)+1
    # min_degree = int(random.random() * 10)+1
    # min_community_size = number_of_point * random.random() * 0.3
    # mixing_parameter = random.random() * 0.15  # 混合参数
    number_of_point = 50  # 节点数
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 3  # 社区大小幂律指数
    average_degree = 6
    min_degree = 2
    min_community_size = 5
    mixing_parameter = 0.1  # 混合参数
    # 生成图
    G, true_communities = create_graph(
        number_of_point,
        min_community_size,
        degree_exponent,
        community_size_exponent,
        average_degree,
        min_degree,
        mixing_parameter,
        seed=72,
    )

    algorithmDealer = AlgorithmDealer()
    lr_algorithm = Leiden_Rare()
    leiden = Leiden()
    # leidenP = LeidenP()
    results = algorithmDealer.run(
        [lr_algorithm, leiden], G, num_clusters=len(true_communities)
    )
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

        CommunityComparator(communities, true_communities).run()

    print(1)
