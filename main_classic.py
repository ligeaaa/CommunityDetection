#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import torch

from algorithm.DL.GCN.gcn import GCN_train_and_evaluate
from algorithm.classic.SBM import sbm_algorithm
from algorithm.classic.louvain import louvain_algorithm
from algorithm.classic.spectral_clustering import spectral_clustering_algorithm
from common.util.data_reader.AmericanFootball import AmericanCollegeFootball
from common.util.data_reader.DBLPDataset import DBLPDataset
from common.util.data_reader.EmailEuCoreDataset import EmailEuCoreDataset
from common.util.data_reader.PoliticalBooksDataset import PolbooksDataset
from common.util.data_reader.ZKClubDataset import ZKClubDataset
from common.util.data_reader.dataset_dealer import Dataset
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集和truthtable（如有）
a = Dataset(ZKClubDataset())
# a = Dataset(PolbooksDataset())
# a = Dataset(AmericanCollegeFootball())
# a = Dataset(EmailEuCoreDataset())

raw_data, truth_table, number_of_community, dataset_name = a.read()

G = nx.Graph()
G.add_edges_from(raw_data)

# 调用算法
# communities = louvain_algorithm(raw_data)
# communities = sbm_algorithm(raw_data, num_blocks=number_of_community)
# communities = spectral_clustering_algorithm(raw_data, num_clusters=number_of_community)
communities = GCN_train_and_evaluate(raw_data, truth_table, device)

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