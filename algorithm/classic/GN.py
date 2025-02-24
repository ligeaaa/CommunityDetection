# !/usr/bin/env python
# coding=utf-8

import networkx as nx
from networkx import Graph

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from algorithm.common.constant.test_data import test_raw_data, test_truth_table
from algorithm.common.util.drawer import draw_communities


class GN(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Girvan-Newman"

    def process(self, G: Graph, **kwargs) -> list:
        """
        1. Calculate betweenness scores for all edges in the network.
        2. Find the edge with the highest score and remove it from the network.
        3. Recalculate betweenness for all remaining edges.
        4. Repeat from step 2.

        Args:
            G (networkx.Graph): An undirected, weighted graph where nodes represent data points
                                and edge weights represent pairwise similar

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
            [1] Newman, M.E.J. and Girvan, M. (2004) ‘Finding and evaluating community structure in networks’,
            Physical Review E, 69(2), p. 026113. Available at: https://doi.org/10.1103/PhysRevE.69.026113.

        """
        # 保存原始图，用于模块度的计算（模块度计算时使用完整网络）
        original_graph = G.copy()
        # 用一个工作图不断移除边
        working_graph = G.copy()

        best_modularity = -1.0
        best_partition = [set(G.nodes())]  # 初始时整个图为一个社区

        # 当图中还有边时持续执行
        while working_graph.number_of_edges() > 0:
            # 计算所有边的最短路径介数
            edge_betweenness = self.compute_edge_betweenness(working_graph)

            # 找到最高的介数值
            max_value = max(edge_betweenness.values())

            # 移除所有介数等于最大值的边
            for edge, value in list(edge_betweenness.items()):
                if value == max_value:
                    working_graph.remove_edge(*edge)

            # 获取当前连通分量作为社区划分
            communities = list(nx.connected_components(working_graph))

            # 根据原始图计算当前划分的模块度 Q
            Q = nx.algorithms.community.modularity(original_graph, communities)

            if Q > best_modularity:
                best_modularity = Q
                best_partition = communities

        # 将每个社区转换为列表后返回
        return [list(community) for community in best_partition]

    def compute_edge_betweenness(self, G: nx.Graph):
        """
        计算图 G 中每条边的最短路径介数

        第一步为广搜，在广搜的过程中做额外操作
        假设从初始点s开始
        1. 设置初始点s的距离$d_s = 0$，权重$w_s = 1$
        2. 对每个与s相邻的点i，设置距离$d_i = d_s + 1 = 1$， 权重$w_i = w_s = 1$
        3. 对每个与i相邻的点j，做以下三个操作中的一个：
                1. 如果j没有设置距离，则$d_j = d_i + 1, w_j = w_i$
                2. 如果j已经设置距离，且$d_j = d_i + 1$，那么$w_j = w_j + w_i$
                3. 如果j已经设置距离，且$d_j < d_i + 1$，那么do nothing
        4. 重复步骤3直到所有结点都被设置距离

        第二步为计算边介数：
        可以开始计算边介数，步骤如下：
        1. 找到所有的"leaf"结点t，i.e., a vertex such that no paths from s to other vertices go though t.
        2. 对于每个与t相邻的结点i，计算i、t之间的边的分数为$w_i / w_t$
        3. 随后开始逐步往初始点s扩展，其它边的分数为：（与它相连接的边的分数的和+1）* $w_i / w_j$
        4. 重复步骤3，直到步骤4

        分别以各个点作为初始点s，将每次计算过程中得到的边介数相加得到最后的边介数

        参数:
            G (networkx.Graph): 无向图

        返回:
            dict: 键为边（以排序元组表示），值为边的介数
        """
        # 初始化每条边的介数，注意对无向图用排序的元组表示边
        betweenness = {tuple(sorted(edge)): 0.0 for edge in G.edges()}

        for s in G.nodes():
            # 初始化 BFS 的数据结构
            S = []  # 用于保存结点的处理顺序（后续反向传播时使用）
            pred = {v: [] for v in G.nodes()}  # 前驱结点列表
            sigma = {v: 0.0 for v in G.nodes()}  # 从 s 到 v 的最短路径数
            dist = {v: -1 for v in G.nodes()}  # 从 s 到 v 的距离，-1 表示未访问

            sigma[s] = 1.0
            dist[s] = 0
            queue = [s]

            # 第一阶段：广度优先搜索 (BFS)
            while queue:
                v = queue.pop(0)
                S.append(v)
                for w in G.neighbors(v):
                    # 如果 w 未访问，则设置距离并加入队列
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # 如果 w 的距离刚好等于 v 的距离 + 1，则找到一条最短路径
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # 初始化每个结点的依赖值
            delta = {v: 0.0 for v in G.nodes()}
            # 第二阶段：反向传播，计算边的介数贡献
            while S:
                w = S.pop()
                for v in pred[w]:
                    # 计算边 (v, w) 的贡献
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    edge = tuple(sorted((v, w)))
                    betweenness[edge] += c
                    delta[v] += c
        # 对于无向图，每条边在两个方向上都被计入，最后除以 2
        for edge in betweenness:
            betweenness[edge] /= 2.0

        return betweenness


if __name__ == "__main__":
    edge_list = test_raw_data
    truth_table = test_truth_table
    G = nx.Graph()
    G.add_edges_from(edge_list)
    algorithmDealer = AlgorithmDealer()
    gn_algorithm = GN()
    results = algorithmDealer.run([gn_algorithm], G)
    communities = results[0].communities
    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos)
    draw_communities(G, pos, communities)
