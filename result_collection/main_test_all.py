#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import time

import networkx as nx
import torch

from algorithm.algorithm_dealer import AlgorithmDealer
from algorithm.classic.GN import GN
from algorithm.classic.RareDetection import RareDetection
from algorithm.classic.leiden import Leiden
from algorithm.classic.louvain import Louvain
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.constant.constant_number import random_seed
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# benchmark生成参数
number_of_point = 150  # 节点数
degree_exponent = 3  # 幂律指数
community_size_exponent = 1.5  # 社区大小幂律指数
average_degree = 5
min_degree = 1
min_community_size = 15
mixing_parameter = 0.1  # 混合参数

# 参数
pkl_directory = r"D:\code\FYP\CommunityDetection\algorithm\common\benchmark\generated_graphs"  # pkl文件读取目录
result_dir = "result/test"  # pkl数据保存目录
os.makedirs(result_dir, exist_ok=True)  # 确保目录存在

if __name__ == "__main__":

    # 生成图
    # from algorithm.common.benchmark.benchmark_graph import create_graph
    # G, true_communities = create_graph(
    #     number_of_point,
    #     min_community_size,
    #     degree_exponent,
    #     community_size_exponent,
    #     average_degree,
    #     min_degree,
    #     mixing_parameter,
    #     seed=random_seed,
    # )

    # 获取目录下所有的 .pkl 文件
    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

    # 读取所有 pkl 文件
    data_list = []
    for file in pkl_files:
        file_path = os.path.join(pkl_directory, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data["title"] = os.path.splitext(file)[0]
            data_list.append(data)

    flag = True

    for data in data_list:
        G = data["graph"]
        num_nodes = G.number_of_nodes()
        # if num_nodes > 100:
        #     continue
        true_communities = data["communities"]
        params = data["params"]
        title = data["title"]

        # if flag and title != "149_point-level2, density-level2, community_size-level3":
        #     continue
        # else:
        #     flag = False

        pos = nx.spring_layout(G, seed=random_seed)
        # 返回结果，包括运行时间，正确率，可视化网络等
        # draw_communities(G, pos, true_communities)

        # 转化 truth_table 的格式
        truth_table = [
            [node, community_id]
            for community_id, nodes in enumerate(true_communities)
            for node in reversed(nodes)
        ]

        # 调用算法
        algorithmDealer = AlgorithmDealer()
        louvain_algorithm = Louvain()
        spectral_clustering_algorithm = SpectralCluster()
        GN_algorithm = GN()
        rare_algorithm = RareDetection()
        leiden_algorithm = Leiden()

        results = algorithmDealer.run(
            [leiden_algorithm],
            G,
            num_clusters=len(true_communities),
        )

        for result in results:
            time.sleep(1)
            communities = result.communities
            algorithm_name = result.algorithm_name

            # 计算评估指标
            evaluation = CommunityDetectionMetrics(G, communities, truth_table)
            metrics = evaluation.evaluate()
            metrics["runtime"] = result.runtime

            # 构造结果文件路径
            result_filename = f"{algorithm_name}-{title}.pkl"
            result_filepath = os.path.join(result_dir, result_filename)

            # 保存数据到 .pkl 文件
            save_data = {
                "communities": communities,
                "algorithm_name": algorithm_name,
                "metrics": metrics,
            }

            with open(result_filepath, "wb") as f:
                pickle.dump(save_data, f)

            print(f"Results saved to {result_filepath}")

            # 可视化结果
            # from algorithm.common.util.drawer import draw_communities

            # draw_communities(
            #     G, pos, communities, title=algorithm_name + "-" + title, metrics=metrics
            # )
