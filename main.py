#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: main.py
@date: 2024/10/3 14:47
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
import networkx as nx

from algorithm.Adaptive.Louvain import Louvain_algorithm
from algorithm.fixed_k.AlinkJaccard import Alink_Jaccard_algorithm
from algorithm.fixed_k.GN_k import GN_algorithm_k
from algorithm.fixed_k.Modularity import Modularity_algorithm
from algorithm.fixed_k.Ncut import Ncut_algorithm
from algorithm.fixed_k.Rcut import Rcut_algorithm
from common.util.data_reader.ZKClubDataset import ZKClubDataset
from common.util.dataset_dealer import Dataset
from common.util.data_reader.EmailEuCoreDataset import EmailEuCoreDataset
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics

# 读取数据集和truthtable（如有）
# a = Dataset(ZKClubDataset)
a = Dataset(EmailEuCoreDataset)


raw_data, truth_table, number_of_community = a.read()

# 调用算法
# G, communities = GN_algorithm(raw_data)
# G, communities = Louvain_algorithm(raw_data)
# G, communities = SpectralClustering_algorithm(raw_data)
# G, communities = Alink_Jaccard_algorithm(raw_data, number_of_community)
G, communities = GN_algorithm_k(raw_data, number_of_community)
# G, communities = Modularity_algorithm(raw_data, number_of_community)
# G, communities = Rcut_algorithm(raw_data, number_of_community)
# G, communities = Ncut_algorithm(raw_data, number_of_community)

# 原始图
pos = nx.spring_layout(G)
draw_communities(G, pos)

# 返回结果，包括运行时间，正确率，可视化网络等
draw_communities(G, pos, communities)

evaluation = CommunityDetectionMetrics(G, communities, truth_table)

metrics = evaluation.evaluate()

# 打印评估结果
for metric, value in metrics.items():
    print(f"{metric}: {value}")