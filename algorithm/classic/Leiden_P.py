import random
import networkx as nx
from networkx import Graph
from collections import defaultdict

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.CommunityCompare import CommunityComparator

# from algorithm.common.util.save_pkl import save_pkl_to_temp
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics


class LeidenP(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Leiden_P"
        self.graph_snapshots = []  # 存储每个阶段的G
        self.original_graph = None  # 初始图
        self.version = "v0.02"

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
            # print(f"-------------{iteration}--------------")

            # while True:
            #     print(f"-------------{iteration}--------------")
            #     if not self.move_nodes_fast(G):
            #         break
            self.move_nodes_fast(G)
            self.detect_and_fix_splits(G)

            self.graph_snapshots.append(G.copy())  # 记录每个阶段的G

            current_communities = set(
                nx.get_node_attributes(G, "community_id").values()
            )

            # 终止条件：
            # 1. 若 num_clusters 指定，则社区数小于等于 num_clusters 时终止
            # 2. 若未指定 num_clusters，则当社区不再变化时终止

            if if_draw and pos is not None and truth_table is not None:
                self.draw_current_result(
                    self.original_graph.copy(), pos, iteration, truth_table
                )

            if (
                num_clusters is not None and len(current_communities) <= num_clusters
            ) or current_communities == prev_communities:
                break

            prev_communities = current_communities.copy()

            G = self.aggregate_graph(G)

        return self.get_final_communities()

    def draw_current_result(
        self, G, pos, iteration, truth_table=None, communities=None
    ):
        if communities is None:
            temp_communities = self.get_final_communities()
        else:
            temp_communities = communities
        if truth_table is not None:
            evaluation = CommunityDetectionMetrics(G, temp_communities, truth_table)
            metrics = evaluation.evaluate()
            draw_communities(
                G, pos, temp_communities, title=f"Leiden_P_{iteration}", metrics=metrics
            )
        else:
            draw_communities(G, pos, temp_communities, title=f"Leiden_P_{iteration}")

    def trace_original_nodes(self, current_community_id, G=None, level=None):
        """
        从当前社区ID递归追溯，获取原始图中属于该社区的所有节点。

        参数：
            current_community_id: 当前阶段的社区 ID
            level: 从第几层（graph_snapshots 的索引）开始追溯，如果为 None 默认最后一层

        返回：
            一个列表，包含原始图中属于该社区的所有节点
        """
        temp_graph_snapshots = self.graph_snapshots.copy()
        if G is not None:
            temp_graph_snapshots.append(G)
        current_community_id = int(current_community_id)

        if level is None:
            level = len(temp_graph_snapshots)

        # 起点：当前社区的 ID
        current_ids = {current_community_id}

        # 从当前层逐步回溯到原始图（索引为 0）
        for L in range(level, 0, -1):
            snapshot = temp_graph_snapshots[L - 1]
            next_ids = set()
            for node in snapshot.nodes():
                comm_id = snapshot.nodes[node]["community_id"]
                if int(comm_id) in current_ids:
                    next_ids.add(int(node))
            current_ids = next_ids

        # 此时 current_ids 就是原始图中属于这个社区的节点
        return list(current_ids)

    def get_final_communities(self):
        """
        逐步回溯 graph_snapshots，还原最终的社区结构。
        追踪每一轮合并的社区对应的原始节点。
        """
        final_snapshot = self.graph_snapshots[-1]
        community_map = defaultdict(list)

        # 获取最终图中每个社区的代表节点（合并后的节点）
        for node in final_snapshot.nodes():
            comm_id = final_snapshot.nodes[node]["community_id"]
            community_map[comm_id].append(node)

        final_communities = []

        # 对于每个最终社区，回溯找到原始图中属于该社区的所有节点
        for comm_id, representative_nodes in community_map.items():
            original_nodes = set()
            traced_nodes = self.trace_original_nodes(int(comm_id))
            original_nodes.update(traced_nodes)
            final_communities.append(sorted(list(original_nodes)))

        return final_communities

    def detect_and_fix_splits(self, G):
        """
        检测社区内部是否存在分裂，并重新整理成新的社区。
        """
        community_map = nx.get_node_attributes(G, "community_id")
        refined_communities = {}

        flag = 0

        for comm in set(community_map.values()):
            subgraph_nodes = [node for node, c in community_map.items() if c == comm]
            subgraph = G.subgraph(subgraph_nodes)
            connected_components = list(nx.connected_components(subgraph))

            for idx, component in enumerate(connected_components):
                refined_communities[f"{flag}"] = component
                flag += 1

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

        return new_G

    def move_nodes_fast(self, G):
        """
        通过模块度增益优化移动节点，以优化社区划分。
        全局构建模块度增益矩阵，贪心选取最大模块度增益对，受限于直径约束。
        """
        expected_diameter = (
            self.expected_lfr_diameter(self.original_graph.number_of_nodes() * 2) + 1
        )
        communities = list(set(nx.get_node_attributes(G, "community_id").values()))
        N = len(communities)

        comm_index_map = {comm: idx for idx, comm in enumerate(communities)}
        index_comm_map = {idx: comm for comm, idx in comm_index_map.items()}

        modularity_matrix = [[-float("inf")] * N for _ in range(N)]

        # 构建模块度增益矩阵（行表示节点所属社区，列表示目标社区）
        for node in G.nodes():
            node_comm = G.nodes[node]["community_id"]
            modularity_results = self.get_modularity_results(G, node)
            if not isinstance(modularity_results, dict):
                continue
            for target_comm, gain in modularity_results.items():
                i = comm_index_map[node_comm]
                j = comm_index_map[target_comm]
                modularity_matrix[i][j] = max(modularity_matrix[i][j], gain)

        visited_communities = set()
        visited_pairs = set()
        changed = False

        while True:
            max_gain = -float("inf")
            best_pair = None
            for i in range(N):
                for j in range(N):
                    if (
                        i == j
                        or i in visited_communities
                        or j in visited_communities
                        or (i, j) in visited_pairs
                    ):
                        continue
                    if modularity_matrix[i][j] > max_gain:
                        max_gain = modularity_matrix[i][j]
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            comm_i = index_comm_map[i]
            comm_j = index_comm_map[j]

            nodes_i = self.trace_original_nodes(comm_i, G=G.copy())
            nodes_j = self.trace_original_nodes(comm_j, G=G.copy())
            merged_nodes = list(set(nodes_i + nodes_j))
            subgraph = self.original_graph.subgraph(merged_nodes)

            if (not nx.is_connected(subgraph)) or (
                nx.diameter(subgraph) > expected_diameter
            ):
                visited_pairs.add((i, j))
                visited_pairs.add((j, i))
                continue

            # print(f"Trying merge community {comm_i} into {comm_j}, diameter: {nx.diameter(subgraph)}")

            # 修改 comm_i 社区内所有节点的 community_id 为 comm_j
            for node in G.nodes():
                if G.nodes[node]["community_id"] == comm_i:
                    G.nodes[node]["community_id"] = comm_j

            changed = True

            # 标记两个社区为已访问
            visited_communities.add(i)
            visited_communities.add(j)

        return changed

    def get_modularity_results(self, G, node):
        """
        计算节点加入不同社区的模块度增益，并选择最优社区。
        """
        modularity_results = {}
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return G.nodes[node]["community_id"]
        # current_community_id = G.nodes[node]["community_id"]
        best_community_id = G.nodes[node]["community_id"]

        for neighbor in neighbors:
            neighbor_community_id = G.nodes[neighbor]["community_id"]

            # current_community = self.trace_original_nodes(current_community_id)
            # target_community = self.trace_original_nodes(neighbor_community_id)
            # num = len(current_community) + len(target_community)

            if neighbor_community_id != best_community_id:
                gain = self.calculate_modularity_gain(G, node, neighbor_community_id)
                # gain *= (1 / num**2)
                # 选择一个符合直径期望的最优社区
                if gain > 0:
                    if neighbor_community_id in modularity_results.keys():
                        modularity_results[neighbor_community_id] = max(
                            modularity_results[neighbor_community_id], gain
                        )
                    else:
                        modularity_results[neighbor_community_id] = gain

        return modularity_results

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
        改进版本：社区大小 n 对应的预期直径，单调递增。
        使用经验公式：a * log(n) + b
        """
        import numpy as np

        a = 0.61
        b = 1.58
        return a * np.log(n) + b


if __name__ == "__main__":
    # edge_list = test_raw_data
    # truth_table = test_truth_table
    # G = nx.Graph()
    # G.add_edges_from(edge_list)
    # 参数设定
    # import random
    # random.seed(52)
    # number_of_point = int(random.random() * 300)  # 节点数
    # degree_exponent = 3  # 幂律指数
    # community_size_exponent = random.random()*2+1  # 社区大小幂律指数
    # average_degree = int(random.random() * 5)+1
    # min_degree = int(random.random() * 10)+1
    # min_community_size = number_of_point * random.random() * 0.3
    # mixing_parameter = random.random() * 0.15  # 混合参数

    number_of_point = 200  # 节点数
    degree_exponent = 3  # 幂律指数
    community_size_exponent = 3  # 社区大小幂律指数
    average_degree = 6
    min_degree = 2
    min_community_size = 10
    mixing_parameter = 0.1  # 混合参数

    # number_of_point = 50  # 节点数
    # degree_exponent = 3  # 幂律指数
    # community_size_exponent = 3  # 社区大小幂律指数
    # average_degree = 4
    # min_degree = 2
    # min_community_size = 3
    # mixing_parameter = 0.1  # 混合参数

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

    for idx, community in enumerate(communities):
        subgraph = G.subgraph(community)
        if nx.is_connected(subgraph):
            print(
                f"Community {idx}, size {len(community)}, diameter = {nx.diameter(subgraph)}"
            )
        else:
            print(f"Community {idx} is disconnected!")

    pos = nx.spring_layout(G, seed=42)
    # draw_communities(G, pos)
    draw_communities(G, pos, communities)

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
    # save_pkl_to_temp(d, "typical_graph")

    # edge_list = test_raw_data
    # truth_table = test_truth_table
    # G = nx.Graph()
    # G.add_edges_from(edge_list)
    # pos = nx.spring_layout(G, seed=42)
    # draw_communities(G, pos)
    # algorithmDealer = AlgorithmDealer()
    # louvain_algorithm = LeidenP()
    # results = algorithmDealer.run([louvain_algorithm], G, num_clusters=2)
    # communities = results[0].communities
    # draw_communities(G, pos, communities)
