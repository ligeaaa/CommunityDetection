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

from algorithm.GN import GN_algorithm
from common.util.dataset_dealer import Dataset
from common.util.data_reader.EmailEuCoreDataset import EmailEuCoreDataset
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics

# 读取数据集和truthtable（如有）
a = Dataset(EmailEuCoreDataset)
raw_data, truth_table = a.read()

# 调用算法
G, communities = GN_algorithm(raw_data)

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