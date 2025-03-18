import networkx as nx
from networkx import Graph

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from algorithm.classic.louvain import Louvain
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.drawer import draw_communities


class RareDetection(Algorithm):
    def __init__(self, N_percent=20, M_percent=40, remove_count=5):
        super().__init__()
        self.algorithm_name = "RareDetection"
        self.N_percent = N_percent / 100  # 转换为比例
        self.M_percent = M_percent / 100
        self.remove_count = remove_count

    def process(self, G: Graph, num_clusters=None, seed=42, **kwargs) -> list:
        original_graph = G.copy()
        degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)

        # 选取度数在前N%~前M%的点集合P
        num_nodes = G.number_of_nodes()
        N_idx = int(num_nodes * self.N_percent)
        M_idx = int(num_nodes * self.M_percent)
        P = [node for node, _ in degree_sorted[N_idx:M_idx]]

        # 计算主流度（这里使用共同邻居数量作为主流度的近似指标）
        mainstream_scores = {
            node: sum(
                len(set(G.neighbors(node)) & set(G.neighbors(neigh)))
                for neigh in G.neighbors(node)
            )
            for node in P
        }

        # 选取主流度最低的 `remove_count` 个点作为“非主流点”
        # TODO 应该不会产生孤立点
        non_mainstream_nodes = sorted(mainstream_scores, key=mainstream_scores.get)[
            : self.remove_count
        ]

        # 找到这些点的邻居，并从图中移除
        removed_nodes = set(non_mainstream_nodes)
        for node in non_mainstream_nodes:
            removed_nodes.update(set(G.neighbors(node)))
        G.remove_nodes_from(removed_nodes)

        # 运行 Louvain 算法
        algorithmDealer = AlgorithmDealer()
        louvain_algorithm = Louvain()
        results = algorithmDealer.run(
            [louvain_algorithm],
            G,
            seed=seed,
            whether_format_result=False,
            num_clusters=num_clusters,
        )
        communities = results[0].communities
        # pos = nx.spring_layout(original_graph, seed=42)
        # draw_communities(original_graph, pos, communities)

        # 重新加入被移除的点，并将其设为独立社区
        for node in removed_nodes:
            communities.append([node])

        # 重新编号社区 ID
        community_mapping = {}
        new_id = 0
        for idx, community in enumerate(communities):
            for node in community:
                community_mapping[node] = idx

        # 重新编号未分类的点
        for node in original_graph.nodes():
            if node not in community_mapping:
                community_mapping[node] = len(communities) + new_id
                new_id += 1

        # 在新的图上再次运行 Louvain
        new_G = original_graph.copy()
        for node in new_G.nodes:
            new_G.nodes[node]["community_id"] = community_mapping[node]

        algorithmDealer = AlgorithmDealer()
        louvain_algorithm = Louvain()
        results = algorithmDealer.run(
            [louvain_algorithm],
            new_G,
            whether_init=False,
            seed=seed,
            num_clusters=num_clusters,
        )
        final_communities = results[0].communities

        return final_communities


if __name__ == "__main__":
    # 参数设定
    number_of_point = 200  # 节点数
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 3  # 社区大小幂律指数
    average_degree = 6
    min_degree = 2
    min_community_size = 15
    mixing_parameter = 0.1  # 混合参数

    # 生成图
    G, true_communities = create_graph(
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
    # draw_communities(G, pos, true_communities)
    algorithmDealer = AlgorithmDealer()
    rare_algorithm = RareDetection()
    results = algorithmDealer.run([rare_algorithm], G.copy(), num_clusters=6)
    communities = results[0].communities

    draw_communities(G, pos, communities)
