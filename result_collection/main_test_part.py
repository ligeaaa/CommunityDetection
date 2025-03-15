#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import re
import time

import networkx as nx
import torch

from algorithm.algorithm_dealer import AlgorithmDealer
from algorithm.classic.GN import GN
from algorithm.classic.louvain import Louvain
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.constant.constant_number import random_seed
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
pkl_directory = r"D:\code\FYP\CommunityDetection\algorithm\common\benchmark\generated_graphs"  # pkl文件读取目录
result_dir = "result/test"  # pkl数据保存目录
os.makedirs(result_dir, exist_ok=True)  # 确保目录存在


def read_data(
    pkl_directory,
    file_id=None,
    point_level=None,
    density_level=None,
    community_size_level=None,
):
    """
    读取目录下所有的 .pkl 文件，并根据筛选条件进行过滤。

    :param pkl_directory: 存储结果的目录路径
    :param file_id: 需要筛选的文件 ID（如 '0', '1', '58'），默认为 None（不过滤）
    :param point_level: 需要筛选的 point-level（如 'level1', 'level2'），默认为 None（不过滤）
    :param density_level: 需要筛选的 density-level（如 'level1', 'level2'），默认为 None（不过滤）
    :param community_size_level: 需要筛选的 community-size-level（如 'level1', 'level2'），默认为 None（不过滤）

    :return: 筛选后的数据列表
    """
    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

    filtered_files = []
    for file in pkl_files:
        match = re.match(
            r"(\d+)_point-(level\d), density-(level\d), community_size-(level\d)\.pkl",
            file,
        )
        if match:
            file_id_match, point_lvl, density_lvl, community_size_lvl = match.groups()

            # 进行筛选
            if (
                (file_id and file_id != file_id_match)
                or (point_level and point_level != point_lvl)
                or (density_level and density_level != density_lvl)
                or (community_size_level and community_size_level != community_size_lvl)
            ):
                continue

            filtered_files.append(file)

    # 读取筛选后的 pkl 文件
    data_list = []
    for file in filtered_files:
        file_path = os.path.join(pkl_directory, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data["title"] = os.path.splitext(file)[0]  # 存储文件名
            data_list.append(data)

    return data_list


if __name__ == "__main__":

    data_list = read_data(
        pkl_directory,
        # file_id="0",
        point_level="level2",
        density_level="level3",
        community_size_level="level1",
    )

    for data in data_list:
        G = data["graph"]
        num_nodes = G.number_of_nodes()
        true_communities = data["communities"]
        params = data["params"]
        title = data["title"]

        pos = nx.spring_layout(G, seed=random_seed)
        # 返回结果，包括运行时间，正确率，可视化网络等
        draw_communities(G, pos, true_communities)

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

        results = algorithmDealer.run(
            [louvain_algorithm, spectral_clustering_algorithm, GN_algorithm],
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
            # draw_communities(G, pos, communities, title=algorithm_name + "-" + title, metrics=metrics)
