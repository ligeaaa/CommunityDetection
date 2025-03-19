#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import re


def read_data(
    result_dir,
    algorithm=None,
    point_level=None,
    density_level=None,
    community_size_level=None,
):
    """
    读取目录下所有的 .pkl 文件，并根据筛选条件进行过滤。

    :param result_dir: 存储结果的目录路径
    :param algorithm: 需要筛选的算法名（如 'Girvan-Newman', 'SpectralCluster', 'Louvain', 'Leiden_Rarev0.01'），默认为 None（不过滤）
    :param point_level: 需要筛选的 point-level（如 'level1', 'level2'），默认为 None（不过滤）
    :param density_level: 需要筛选的 density-level（如 'level1', 'level2'），默认为 None（不过滤）
    :param community_size_level: 需要筛选的 community-size-level（如 'level1', 'level2'），默认为 None（不过滤）

    :return: 筛选后的数据列表
    """
    pkl_files = [f for f in os.listdir(result_dir) if f.endswith(".pkl")]

    filtered_files = []
    for file in pkl_files:
        match = re.match(
            r"([\w\-.]+)-(\d+)_point-(level\d), density-(level\d), community_size-(level\d)\.pkl",
            file,
        )
        if match:
            algo_name, point, point_lvl, density_lvl, community_size_lvl = (
                match.groups()
            )

            # 进行筛选
            if (
                (algorithm and algorithm != algo_name)
                or (point_level and point_level != point_lvl)
                or (density_level and density_level != density_lvl)
                or (community_size_level and community_size_level != community_size_lvl)
            ):
                continue

            filtered_files.append(file)

    # 读取筛选后的 pkl 文件
    data_list = []
    for file in filtered_files:
        file_path = os.path.join(result_dir, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data["title"] = os.path.splitext(file)[0]  # 存储文件名
            data_list.append(data)

    return data_list
