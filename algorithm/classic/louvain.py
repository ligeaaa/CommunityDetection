#!/usr/bin/env python
# coding=utf-8
import random

import networkx as nx
from networkx import Graph

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from algorithm.common.constant.test_data import test_raw_data, test_truth_table
from algorithm.common.util.drawer import draw_communities


class Louvain(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Louvain"
        self.graph_snapshots = []  # 存储每个阶段的G
        self.original_graph = None  # 初始图

    def process(
        self, G: Graph, num_clusters=None, seed=42, whether_init=True, **kwargs
    ) -> list:
        """
        First, we assign a different community to each node of the network.

        Then, for each node i we consider the neighbours j of i, and we evaluate the gain of modularity
        that would take place by removing i from its community and by placing it in the community of j.

        This process is applied repeatedly and sequentially for all nodes until no further improvement
        can be achieved and the first phase is then complete.

        The second phase of the algorithm consists in building a new network whose nodes are now the
        communities found during the first phase. To do so, the weights of the links between the new
        nodes are given by the sum of the weight of the links between nodes in the corresponding two communities

        Args:
            G (networkx.Graph): An undirected, weighted graph where nodes represent data points
                                and edge weights represent pairwise similar
            seed: random seed, default 42

        Returns:
            list: A list of communities, where each community is a list of node IDs.
                  Example output:
                  ```
                  [[0, 1, 2, 3, 4],  # Community 0
                   [5, 6, 7, 8, 9]]  # Community 1
                  ```
                  This means nodes 0,1,2,3,4 belong to community 0,
                  and nodes 5,6,7,8,9 belong to community 1.

        References:
            [1] Blondel, V.D. et al. (2008) ‘Fast unfolding of communities in large networks’,
            Journal of Statistical Mechanics: Theory and Experiment, 2008(10), p. P10008.
            Available at: https://doi.org/10.1088/1742-5468/2008/10/P10008.

        """

        self.original_graph = G.copy()
        if whether_init:
            self.G = self.init_G(G)
        else:
            self.G = G
        self.graph_snapshots.append(self.G.copy())  # 记录初始图
        iter_time = 0
        while True:
            iter_time += 1
            mod_inc = self.first_phase(seed)

            if mod_inc:
                self.second_phase()
                self.graph_snapshots.append(self.G.copy())  # 记录每个阶段的G
                current_communities = self.get_communities()
                if (
                    num_clusters is not None
                    and len(current_communities) <= num_clusters
                ):
                    break  # 达到目标社区数量时停止
            else:
                break
        return self.get_final_communities()

    def first_phase(self, seed):
        mod_inc = False
        random.seed(seed)
        visit_sequence = list(self.G.nodes())
        # 随机点的顺序
        random.shuffle(visit_sequence)
        while True:
            can_stop = True
            for node in visit_sequence:
                community_id = self.G.nodes[node].get("community_id")
                best_community = community_id
                max_modularity_gain = 0
                for neighbor in self.G.neighbors(node):
                    neighbor_community = self.G.nodes[neighbor].get("community_id")
                    if neighbor_community != community_id:
                        gain = self.calculate_modularity_gain(node, neighbor_community)
                        if gain > max_modularity_gain:
                            max_modularity_gain = gain
                            best_community = neighbor_community
                if best_community != community_id:
                    self.G.nodes[node]["community_id"] = best_community
                    mod_inc = True
                    can_stop = False
            if can_stop:
                break
        return mod_inc

    def calculate_modularity_gain(self, node, target_community):
        """
        计算将节点移动到目标社区后的模块度增益。

        Args:
            node: 要移动的节点
            target_community: 目标社区 ID

        Returns:
            模块度增益值
        """
        current_community = self.G.nodes[node].get("community_id")
        original_modularity = nx.algorithms.community.modularity(
            self.G, self.get_communities()
        )
        self.G.nodes[node]["community_id"] = target_community
        new_modularity = nx.algorithms.community.modularity(
            self.G, self.get_communities()
        )
        self.G.nodes[node]["community_id"] = current_community
        return new_modularity - original_modularity

    def second_phase(self):
        new_graph = nx.Graph()
        community_mapping = {}
        node_to_community = {}
        for node in self.G.nodes():
            community_id = self.G.nodes[node]["community_id"]
            if community_id not in community_mapping:
                community_mapping[community_id] = []
            community_mapping[community_id].append(node)
            node_to_community[node] = community_id
        for community_id, nodes in community_mapping.items():
            new_graph.add_node(community_id, merged_nodes=nodes)
        for u, v in self.G.edges():
            cu = node_to_community[u]
            cv = node_to_community[v]
            if cu != cv:
                weight = (
                    new_graph[cu][cv]["weight"] + 1 if new_graph.has_edge(cu, cv) else 1
                )
                new_graph.add_edge(cu, cv, weight=weight)
        self.G = self.init_G(new_graph)

    def init_G(self, G: Graph):
        for node in G.nodes():
            G.nodes[node]["community_id"] = node
        return G

    def get_communities(self):
        communities = {}
        for node in self.G.nodes():
            community_id = self.G.nodes[node]["community_id"]
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        return [sorted(nodes) for nodes in communities.values()]

    def get_final_communities(self):
        final_communities = {}

        # 追踪每个节点的最终归属社区
        for snapshot in reversed(self.graph_snapshots):
            if snapshot.number_of_nodes() == 1:
                continue
            for node in snapshot.nodes():
                if "merged_nodes" in snapshot.nodes[node]:
                    for original_node in snapshot.nodes[node]["merged_nodes"]:
                        if original_node not in final_communities:
                            final_communities[original_node] = node

        def find_root(node):
            """使用路径压缩方法查找根社区，避免死循环"""
            visited = set()
            while node in final_communities and final_communities[node] != node:
                if node in visited:
                    final_communities[node] = node  # 发现环，打破并指向自身
                    break
                visited.add(node)
                next_node = final_communities[node]
                final_communities[node] = final_communities.get(next_node, next_node)
                node = next_node
            return node

        # 修正所有节点的社区归属，避免环
        for node in list(final_communities.keys()):
            final_communities[node] = find_root(node)

        # 重新整理社区结构
        grouped_communities = {}
        for node, community in final_communities.items():
            if community not in grouped_communities:
                grouped_communities[community] = []
            grouped_communities[community].append(node)

        return [sorted(nodes) for nodes in grouped_communities.values()]


if __name__ == "__main__":
    edge_list = test_raw_data
    truth_table = test_truth_table
    G = nx.Graph()
    G.add_edges_from(edge_list)
    algorithmDealer = AlgorithmDealer()
    louvain_algorithm = Louvain()
    results = algorithmDealer.run([louvain_algorithm], G, num_clusters=2)
    communities = results[0].communities
    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos)
    draw_communities(G, pos, communities)
