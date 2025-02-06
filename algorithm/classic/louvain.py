#!/usr/bin/env python
# coding=utf-8
import community as community_louvain
import networkx as nx
from networkx import Graph

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from algorithm.common.util.drawer import draw_communities


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

        best_communities = [
            sorted(nodes) for nodes in communities.values()
        ]  # 按节点ID排序

        return best_communities


if __name__ == "__main__":
    # 示例输入：边的列表
    edge_list = [
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 4],
        [3, 5],
        [4, 5],
        [5, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [7, 8],
        [7, 9],
        [7, 10],
        [8, 9],
        [8, 10],
        [9, 10],
    ]
    truth_table = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

    # 调用 Louvain 算法并返回图和社区结果
    G = nx.Graph()
    G.add_edges_from(edge_list)
    algorithmDealer = AlgorithmDealer()
    louvain_algorithm = Louvain()

    # 可视化结果
    results = algorithmDealer.run([louvain_algorithm], G)
    communities = results[0].communities

    pos = nx.spring_layout(G)
    draw_communities(G, pos)

    # 返回结果，包括运行时间，正确率，可视化网络等
    draw_communities(G, pos, communities)
