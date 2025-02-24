#!/usr/bin/env python
# coding=utf-8

import networkx as nx
import torch

from algorithm.algorithm_dealer import AlgorithmDealer
from algorithm.classic.GN import GN
from algorithm.classic.louvain import Louvain
from algorithm.classic.SBM import SBM
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.util.data_reader.AmazonDataset import AmazonDataset
from algorithm.common.util.data_reader.AmericanFootball import AmericanCollegeFootball
from algorithm.common.util.data_reader.CoraDataset import CoraDataset
from algorithm.common.util.data_reader.dataset_dealer import Dataset
from algorithm.common.util.data_reader.EmailEuCoreDataset import EmailEuCoreDataset
from algorithm.common.util.data_reader.PoliticalBooksDataset import PolbooksDataset
from algorithm.common.util.data_reader.ZKClubDataset import ZKClubDataset
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 读取数据集和truthtable（如有）
# a = Dataset(ZKClubDataset())
a = Dataset(PolbooksDataset())
# a = Dataset(AmericanCollegeFootball())
# a = Dataset(EmailEuCoreDataset())
# a = Dataset(CoraDataset())
# a = Dataset(AmazonDataset())

raw_data, truth_table, num_clusters, dataset_name = a.read()

G = nx.Graph()
G.add_edges_from(raw_data)

# 调用算法
algorithmDealer = AlgorithmDealer()
louvain_algorithm = Louvain()
sbm_algorithm = SBM()
spectral_clustering_algorithm = SpectralCluster()
GN_algorithm = GN()
# accuracy, nmi, mod, runtime = GCN_train_and_evaluate(raw_data, truth_table, device)
# communities = GCN_train_unsupervised(raw_data, device, epochs=1000, learning_rate=0.01, margin=1.0)

# results = algorithmDealer.run([louvain_algorithm], G)
# results = algorithmDealer.run([sbm_algorithm], G, num_clusters=num_clusters)
results = algorithmDealer.run([GN_algorithm], G)
# results = algorithmDealer.run([spectral_clustering_algorithm], G, num_clusters=num_clusters)
communities = results[0].communities


# 原始图
pos = nx.spring_layout(G, seed=42)
draw_communities(G, pos)

# 返回结果，包括运行时间，正确率，可视化网络等
draw_communities(G, pos, communities)

evaluation = CommunityDetectionMetrics(G, communities, truth_table)

metrics = evaluation.evaluate()

# 打印评估结果
for metric, value in metrics.items():
    print(f"{metric}: {value}")
