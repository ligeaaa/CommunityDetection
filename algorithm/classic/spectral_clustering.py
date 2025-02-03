#!/usr/bin/env python
# coding=utf-8
import networkx as nx
from networkx import Graph
from sklearn.cluster import SpectralClustering

from algorithm.algorithm_dealer import Algorithm


class SpectralCluster(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "SBM"

    def process(self, G: Graph, **kwargs):
        # 获取邻接矩阵
        adj_matrix = nx.to_numpy_array(G)

        # 使用谱聚类算法进行社区划分
        spectral_model = SpectralClustering(n_clusters=kwargs['num_clusters'], affinity='precomputed', random_state=42)
        labels = spectral_model.fit_predict(adj_matrix)

        # 检测节点编号是否从 0 开始或 1 开始
        min_node = min(G.nodes())

        # 根据标签进行社区划分，调整节点编号
        communities = {}
        for node, label in zip(G.nodes(), labels):  # 使用 G.nodes() 中的实际节点编号
            if label not in communities:
                communities[label] = []
            communities[label].append(node)

        # 确保社区划分没有遗漏节点
        all_nodes = set(G.nodes())  # 图中所有节点
        assigned_nodes = set(node for community in communities.values() for node in community)  # 所有已分配的节点
        unassigned_nodes = all_nodes - assigned_nodes  # 没有被分配的节点

        # 仅当确实有未分配节点时，将它们归为一个单独的社区
        if unassigned_nodes:
            communities[len(communities)] = list(unassigned_nodes)

        best_communities = [sorted(nodes) for nodes in communities.values()]

        return best_communities

