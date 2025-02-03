#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import community as community_louvain
from networkx import Graph

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from common.util.decorator import time_record
from common.util.drawer import draw_communities
from common.util.result_evaluation import CommunityDetectionMetrics


class Louvain(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Louvain"

    def process(self, G: Graph, **kwargs):
        # 使用Louvain算法进行社区划分
        partition = community_louvain.best_partition(G)

        # 将节点根据其社区编号进行分组
        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        best_communities = [sorted(nodes) for nodes in communities.values()]  # 按节点ID排序

        return best_communities


if __name__ == '__main__':
    # 示例输入：边的列表
    edge_list = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3], [2, 4], [4, 6], [6, 7]]
    truth_table = [[7, 0], [6, 0], [4, 1], [5, 1], [3, 1], [2, 1], [1, 1]]

    # 调用 Louvain 算法并返回图和社区结果
    G = nx.Graph()
    G.add_edges_from(edge_list)
    algorithmDealer = AlgorithmDealer()
    louvain_algorithm = Louvain()

    # 可视化结果
    results = algorithmDealer.process([louvain_algorithm], G)
    communities = results[0].communities

    pos = nx.spring_layout(G)
    draw_communities(G, pos)

    # 返回结果，包括运行时间，正确率，可视化网络等
    draw_communities(G, pos, communities)