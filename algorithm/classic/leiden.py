import random
import networkx as nx
from networkx import Graph
from collections import defaultdict

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.drawer import draw_communities


class Leiden(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Leiden"
        self.graph_snapshots = []  # 存储每个阶段的G
        self.original_graph = None  # 初始图

    def process(
        self, G: Graph, whether_init=True, seed=42, num_clusters=None, **kwargs
    ) -> list:
        """
        主要的 Leiden 算法入口，执行社区发现。
        若分类后的社区数小于等于 num_clusters，则提前终止。
        """
        random.seed(seed)
        self.original_graph = G.copy()

        # 初始化每个节点的社区编号
        for node in G.nodes():
            G.nodes[node]["community_id"] = node

        iteration = 0
        while True:
            iteration += 1

            self.move_nodes_fast(G)
            self.graph_snapshots.append(G.copy())  # 记录每个阶段的G

            if (
                num_clusters is not None
                and len(set(nx.get_node_attributes(G, "community_id").values()))
                <= num_clusters
            ):
                break

            self.aggregate_graph(G)

        return self.get_final_communities()

    def get_final_communities(self):
        """
        逐步回溯 `graph_snapshots`，还原最终的社区结构。
        """
        final_communities = {}

        # 初始化最终社区映射
        last_snapshot = self.graph_snapshots[-1]
        for node in last_snapshot.nodes():
            final_communities[node] = node

        # 逆向追踪社区合并过程
        for snapshot in reversed(self.graph_snapshots[:-1]):
            new_mapping = {}
            for node in snapshot.nodes():
                community_id = snapshot.nodes[node]["community_id"]
                final_community = final_communities.get(community_id, community_id)
                new_mapping[node] = final_community
            final_communities = new_mapping

        # 组织最终社区结构
        communities = defaultdict(set)
        for node, comm_id in final_communities.items():
            communities[comm_id].add(node)
        return [sorted(list(nodes)) for nodes in communities.values()]

    def move_nodes_fast(self, G):
        """
        通过贪心策略移动节点，以优化社区划分。
        """
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for node in nodes:
            best_community = self.best_community(G, node)
            if (
                best_community is not None
                and best_community != G.nodes[node]["community_id"]
            ):
                G.nodes[node]["community_id"] = best_community

    def best_community(self, G, node):
        """
        计算节点加入不同社区的收益，并选择最优社区。
        """
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return G.nodes[node]["community_id"]

        community_scores = defaultdict(float)
        for neighbor in neighbors:
            neighbor_community = G.nodes[neighbor]["community_id"]
            community_scores[neighbor_community] += G[node][neighbor].get("weight", 1)

        return max(
            community_scores,
            key=community_scores.get,
            default=G.nodes[node]["community_id"],
        )

    def aggregate_graph(self, G):
        """
        构建新的聚合图，每个社区变成一个节点，边权重为社区间边的总和。
        """
        new_G = nx.Graph()
        community_map = nx.get_node_attributes(G, "community_id")

        for node1, node2, data in G.edges(data=True):
            comm1, comm2 = community_map[node1], community_map[node2]
            if comm1 != comm2:
                if new_G.has_edge(comm1, comm2):
                    new_G[comm1][comm2]["weight"] += data.get("weight", 1)
                else:
                    new_G.add_edge(comm1, comm2, weight=data.get("weight", 1))

        for comm in set(community_map.values()):
            new_G.add_node(comm, community_id=comm)

        G.clear()
        G.add_edges_from(new_G.edges(data=True))
        nx.set_node_attributes(
            G, {node: node for node in new_G.nodes()}, "community_id"
        )


if __name__ == "__main__":
    # edge_list = test_raw_data
    # truth_table = test_truth_table
    # G = nx.Graph()
    # G.add_edges_from(edge_list)
    # 参数设定
    number_of_point = 200  # 节点数
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 3  # 社区大小幂律指数
    average_degree = 6
    min_degree = 2
    min_community_size = 15
    mixing_parameter = 0.1  # 混合参数

    # 生成图
    G, truth_table = create_graph(
        number_of_point,
        min_community_size,
        degree_exponent,
        community_size_exponent,
        average_degree,
        min_degree,
        mixing_parameter,
        seed=53,
    )
    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos, truth_table)
    algorithmDealer = AlgorithmDealer()
    louvain_algorithm = Leiden()
    results = algorithmDealer.run([louvain_algorithm], G, num_clusters=len(truth_table))
    communities = results[0].communities
    pos = nx.spring_layout(G, seed=42)
    # draw_communities(G, pos)
    draw_communities(G, pos, communities)
