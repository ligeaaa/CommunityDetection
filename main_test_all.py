#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import torch

from algorithm.algorithm_dealer import AlgorithmDealer
from algorithm.classic.GN import GN
from algorithm.classic.louvain import Louvain
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.constant.constant_number import random_seed
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
number_of_point = 150  # 节点数
degree_exponent = 3  # 幂律指数
community_size_exponent = 1.5  # 社区大小幂律指数
average_degree = 5
min_degree = 1
min_community_size = 15
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
    seed=random_seed,
)
pos = nx.spring_layout(G, seed=random_seed)
# 返回结果，包括运行时间，正确率，可视化网络等
draw_communities(G, pos, true_communities)

# 转化truth_table的格式
result = []
for community_id, nodes in enumerate(true_communities):
    for node in reversed(nodes):  # 反向遍历节点
        result.append([node, community_id])
truth_table = result


# 调用算法
algorithmDealer = AlgorithmDealer()
louvain_algorithm = Louvain()
spectral_clustering_algorithm = SpectralCluster()
GN_algorithm = GN()

results = algorithmDealer.run(
    [louvain_algorithm, spectral_clustering_algorithm, GN_algorithm],
    G,
    num_clusters=len(true_communities),
)

for result in results:
    communities = result.communities
    algorithm_name = result.algorithm_name
    # 返回结果，包括运行时间，正确率，可视化网络等
    evaluation = CommunityDetectionMetrics(G, communities, truth_table)
    metrics = evaluation.evaluate()
    metrics["runtime"] = result.runtime
    draw_communities(
        G, pos, communities, algorithm_name=algorithm_name, metrics=metrics
    )

    # 打印评估结果
    print("---" + algorithm_name + "---")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
