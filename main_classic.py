#!/usr/bin/env python
# coding=utf-8
import networkx as nx

from algorithm.classic.louvain import louvain_algorithm
from common.util.data_reader.DBLPDataset import DBLPDataset
from common.util.data_reader.ZKClubDataset import ZKClubDataset
from common.util.data_reader.dataset_dealer import Dataset
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics

# 读取数据集和truthtable（如有）
a = Dataset(ZKClubDataset())
# a = Dataset(EmailEuCoreDataset())
# a = Dataset(DBLPDataset())


raw_data, truth_table, number_of_community, dataset_name = a.read()

# 调用算法
G, communities = louvain_algorithm(raw_data)
# G, communities = sbm_algorithm(raw_data, num_blocks=number_of_community)
# G, communities = spectral_clustering_algorithm(raw_data, num_clusters=number_of_community)

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