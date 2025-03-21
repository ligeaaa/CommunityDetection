import random
import networkx as nx
import numpy as np
from networkx import Graph
from collections import defaultdict

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.CommunityCompare import CommunityComparator
from algorithm.common.util.data_reader.save_pkl import save_pkl_to_temp
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics


class LeidenP(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Leiden_P"
        self.graph_snapshots = []  # 存储每个阶段的G
        self.original_graph = None  # 初始图
        self.version = "v0.01"

    def process(
        self,
        G: Graph,
        whether_init=True,
        seed=42,
        num_clusters=None,
        if_draw=False,
        pos=None,
        truth_table=None,
        **kwargs,
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
        prev_communities = None
        while True:

            iteration += 1
            self.move_nodes_fast(G)
            self.detect_and_fix_splits(G)
            self.graph_snapshots.append(G.copy())  # 记录每个阶段的G
            if if_draw and pos is not None and truth_table is not None:
                self.draw_current_result(
                    self.original_graph, pos, iteration, truth_table
                )

            current_communities = set(
                nx.get_node_attributes(G, "community_id").values()
            )

            # 终止条件：
            # 1. 若 num_clusters 指定，则社区数小于等于 num_clusters 时终止
            # 2. 若未指定 num_clusters，则当社区不再变化时终止
            if (
                num_clusters is not None and len(current_communities) <= num_clusters
            ) or (num_clusters is None and current_communities == prev_communities):
                break

            prev_communities = current_communities.copy()
            self.aggregate_graph(G)

        return self.get_final_communities()

    def draw_current_result(self, G, pos, iteration, truth_table):
        temp_communities = self.get_final_communities()
        evaluation = CommunityDetectionMetrics(G, temp_communities, truth_table)
        metrics = evaluation.evaluate()
        draw_communities(
            G, pos, temp_communities, title=f"Leiden_P_{iteration}", metrics=metrics
        )

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

    def detect_and_fix_splits(self, G):
        """
        检测社区内部是否存在分裂，并重新整理成新的社区。
        """
        community_map = nx.get_node_attributes(G, "community_id")
        refined_communities = {}

        for comm in set(community_map.values()):
            subgraph_nodes = [node for node, c in community_map.items() if c == comm]
            subgraph = G.subgraph(subgraph_nodes)
            connected_components = list(nx.connected_components(subgraph))

            for idx, component in enumerate(connected_components):
                refined_communities[f"{comm}_{idx}"] = component

        # 更新 G 中的社区编号
        for new_comm_id, nodes in refined_communities.items():
            for node in nodes:
                G.nodes[node]["community_id"] = new_comm_id

    def aggregate_graph(self, G):
        """
        构建新的聚合图，每个社区变成一个节点，边权重为社区间边的总和。
        """
        new_G = nx.Graph()
        community_map = nx.get_node_attributes(G, "community_id")

        # 构建新的社区图
        for node1, node2, data in G.edges(data=True):
            comm1, comm2 = community_map[node1], community_map[node2]
            if comm1 != comm2:
                if new_G.has_edge(comm1, comm2):
                    new_G[comm1][comm2]["weight"] += data.get("weight", 1)
                else:
                    new_G.add_edge(comm1, comm2, weight=data.get("weight", 1))

        # 添加新的社区节点
        for comm in set(community_map.values()):
            new_G.add_node(comm, community_id=comm)

        # 更新G，确保社区编号正确
        G.clear()
        G.add_edges_from(new_G.edges(data=True))
        nx.set_node_attributes(
            G, {node: node for node in new_G.nodes()}, "community_id"
        )

    def move_nodes_fast(self, G):
        """
        通过模块度增益优化移动节点，以优化社区划分。
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
        计算节点加入不同社区的模块度增益，并选择最优社区。
        """

        def adjusted_modularity_gain(
            G,
            node,
            pre_target_community_nodes,
            lambda_penalty=1.0,
        ):
            """
            根据社区直径变化和预期直径惩罚修正模块度增益。

            参数：
                G: NetworkX 图
                node: 当前尝试移动的节点
                pre_target_community_nodes: 当前目标社区的节点列表（不含 node）
                lambda_penalty: 惩罚项权重参数（默认 1.0）

            返回：
                修正后的模块度增益
            """
            # 目标社区原始直径与预期直径
            # expected_diameter_before = self.expected_lfr_diameter(
            #     len(pre_target_community_nodes)
            # )
            pre_subgraph = G.subgraph(pre_target_community_nodes)
            if nx.is_connected(pre_subgraph):
                actual_diameter_before = nx.diameter(pre_subgraph)
            else:
                return 0

            # 模拟将 node 加入社区后的直径与预期值
            after_nodes = pre_target_community_nodes.copy()
            after_nodes.append(node)
            expected_diameter_after = self.expected_lfr_diameter(len(after_nodes))
            # after_subgraph = G.subgraph(after_nodes)
            if nx.is_connected(pre_subgraph):
                # actual_diameter_after = nx.diameter(after_subgraph)
                ...
            else:
                return 0

            # 惩罚项（可以换其他策略）
            # delta_diameter = (actual_diameter_after - expected_diameter_after) - \
            #                  (actual_diameter_before - expected_diameter_before)

            delta_diameter = expected_diameter_after - actual_diameter_before

            diameter_penalty = lambda_penalty * delta_diameter

            return diameter_penalty

        neighbors = list(G.neighbors(node))
        if not neighbors:
            return G.nodes[node]["community_id"]

        best_community = G.nodes[node]["community_id"]
        max_modularity_gain = 0

        for neighbor in neighbors:
            neighbor_community = G.nodes[neighbor]["community_id"]
            if neighbor_community != best_community:
                gain = self.calculate_modularity_gain(G, node, neighbor_community)

                if gain > 0:
                    pre_target_community_nodes = [
                        n
                        for n in G.nodes
                        if G.nodes[n]["community_id"] == neighbor_community
                    ]
                    if len(pre_target_community_nodes) > 17:
                        diameter_penalty = adjusted_modularity_gain(
                            G,
                            node,
                            pre_target_community_nodes,
                            lambda_penalty=1.0,
                        )
                        gain = gain * diameter_penalty

                # 选择一个符合直径期望的最优社区
                if gain > max_modularity_gain:
                    max_modularity_gain = gain
                    best_community = neighbor_community

        return best_community

    def calculate_modularity_gain(self, G, node, target_community):
        """
        计算将节点移动到目标社区的模块度增益。
        """
        m = sum(data.get("weight", 1) for _, _, data in G.edges(data=True))
        k_i = sum(G[node][neighbor].get("weight", 1) for neighbor in G.neighbors(node))
        sum_in = sum(
            G[neighbor][node].get("weight", 1)
            for neighbor in G.neighbors(node)
            if G.nodes[neighbor]["community_id"] == target_community
        )
        sum_tot = sum(
            G[neighbor][node].get("weight", 1) for neighbor in G.neighbors(node)
        )

        delta_Q = (sum_in + k_i) / (2 * m) - ((sum_tot + k_i) / (2 * m)) ** 2
        return delta_Q

    def expected_lfr_diameter(self, n):
        """
        计算 LFR 预期直径
        """
        return np.log(n) / np.log(np.log(n))


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
    min_community_size = 10
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
    truth_table = [
        [node, community_id]
        for community_id, nodes in enumerate(true_communities)
        for node in reversed(nodes)
    ]
    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos, true_communities)
    algorithmDealer = AlgorithmDealer()
    leidenP = LeidenP()
    results = algorithmDealer.run(
        [leidenP],
        G,
        num_clusters=len(true_communities),
        pos=pos,
        if_draw=True,
        truth_table=truth_table,
    )
    communities = results[0].communities
    pos = nx.spring_layout(G, seed=42)
    # draw_communities(G, pos)
    # draw_communities(G, pos, communities)

    # 计算评估指标
    # 转化 truth_table 的格式
    evaluation = CommunityDetectionMetrics(G, communities, truth_table)
    metrics = evaluation.evaluate()
    metrics["runtime"] = results[0].runtime

    # 可视化结果
    from algorithm.common.util.drawer import draw_communities

    draw_communities(G, pos, communities, title="Leiden_P", metrics=metrics)

    CommunityComparator(communities, true_communities).run()

    d = {"G": G, "communities": communities}
    save_pkl_to_temp(d, "typical_graph")
