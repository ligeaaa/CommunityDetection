import random
from collections import defaultdict

import networkx as nx
from networkx import Graph

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.classic.leiden import Leiden
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.CommunityCompare import CommunityComparator
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics


class Leiden_Rare(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "Leiden_Rare"
        self.version = "v0.02"
        self.graph_snapshots = []  # 存储每个阶段的G
        self.original_graph = None  # 初始图

    def process(
        self,
        G: Graph,
        whether_init=True,
        seed=42,
        num_clusters=2,
        diameter_check_parameter=0.05,
        density_threshold=0.1,
        **kwargs,
    ) -> list:
        """
        使用 Leiden 预分配社区，并对直径过大的社区进行进一步划分（使用谱聚类）。
        """
        random.seed(seed)
        self.original_graph = G.copy()

        # 步骤 1: 使用 Leiden 进行初步社区检测
        algorithmDealer = AlgorithmDealer()
        leiden_algorithm = Leiden()
        results = algorithmDealer.run(
            [leiden_algorithm],
            G,
            num_clusters=num_clusters,
            whether_format_result=True,
        )
        communities = results[0].communities

        # 步骤 2: 根据直径切割社区
        communities_dict = self.split_community(G, communities)

        # **最终返回优化后的社区**
        return [sorted(list(nodes)) for nodes in communities_dict.values()]

    def split_community(self, G, communities):
        """

        Args:
            G:
            communities:

        Returns:
            dict: A dictionary representing community detection results.
                Structure:
                    - Keys (int): Community IDs.
                    - Values (list of int): List of node IDs belonging to the corresponding community.
                Example:
                    {
                        0: [0, 1, 2, 3, 4],  # Community 0 contains nodes 0,1,2,3,4
                        1: [5, 6, 7, 8, 9],  # Community 1 contains nodes 5,6,7,8,9
                        2: [10, 11, 12]      # Community 2 contains nodes 10,11,12
                    }
        """

        def expected_lfr_diameter(n):
            """
            计算 LFR 预期直径
            """
            import numpy as np

            return np.log(n) / np.log(np.log(n))

        def compute_community_diameters(G, communities):
            """
            计算每个社区的直径，并按照直径从大到小排序返回。

            参数:
                G: networkx.Graph - 原始图
                communities: list - 社区列表，每个社区是一个节点 ID 列表

            返回:
                list - [(社区索引, 直径)] 按直径降序排列
            """
            diameters = {}

            for i, community in enumerate(communities):
                subgraph = G.subgraph(community)  # 提取子图

                if nx.is_connected(subgraph):  # 如果连通
                    diameters[i] = nx.diameter(subgraph)
                else:
                    # 对于不连通的图，计算最大连通子图的直径
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                    diameters[i] = nx.diameter(G.subgraph(largest_cc))

            # 按直径降序排序
            sorted_diameters = sorted(
                diameters.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_diameters

        def check_split(G, communities_dict, largest_community_id):
            """
            计算社区子图中所有节点对的最短路径，整理并统计最长最短路径比例。
            """
            # 获取子图
            sub_G = G.subgraph(communities_dict[largest_community_id])

            # 计算所有节点对的最短路径长度
            shortest_path_dict = defaultdict(list)
            path_lengths = dict(nx.all_pairs_shortest_path_length(sub_G))

            for node1 in path_lengths:
                for node2, length in path_lengths[node1].items():
                    if node1 < node2:  # 避免重复存储 (1,2) 和 (2,1)
                        shortest_path_dict[length].append([node1, node2])

            # 按最短路径长度从大到小排序
            sorted_shortest_paths = dict(
                sorted(shortest_path_dict.items(), key=lambda x: x[0], reverse=True)
            )

            # 计算最长的最短路径占比
            max_length = max(sorted_shortest_paths.keys())  # 获取最长的最短路径长度
            total_paths = (
                sub_G.number_of_nodes() * (sub_G.number_of_nodes() - 1) / 2
            )  # 总最短路径数量
            max_length_count = len(
                sorted_shortest_paths[max_length]
            )  # 最长最短路径的数量
            max_length_ratio = max_length_count / total_paths if total_paths > 0 else 0
            if max_length_ratio > 0.0001:
                return True
            else:
                return False

        # 计算预期直径
        expected_diameter = round(
            expected_lfr_diameter(
                G.number_of_nodes() / 2,  # 取一半节点数
            )
        )

        # 计算初始社区直径
        community_diameters = compute_community_diameters(G, communities)

        # 构建社区字典
        communities_dict = {idx: nodes for idx, nodes in enumerate(communities)}

        # 迭代优化
        next_community_id = len(communities_dict)  # 记录新社区编号
        while True:
            # 找到直径最大的社区
            largest_community_id, largest_diameter = max(
                dict(community_diameters).items(), key=lambda x: x[1]
            )

            # 如果所有社区直径都 小于 预期直径，结束
            if largest_diameter <= expected_diameter + 1:
                break

            # 如果不满足条件，则不split，并把该社区从待处理社区中移除
            if not check_split(G, communities_dict, largest_community_id):
                community_diameters.remove((largest_community_id, largest_diameter))
                continue

            # print(
            #     f"⚠️ 社区 {largest_community_id} 直径过大 ({largest_diameter} > {expected_diameter + 1})，进行谱聚类划分"
            # )

            # 获取需要拆分的社区
            largest_community_nodes = communities_dict[largest_community_id]
            subgraph = G.subgraph(largest_community_nodes).copy()

            # 使用谱聚类进行社区拆分
            algorithmDealer = AlgorithmDealer()
            sc_algorithm = SpectralCluster()
            results = algorithmDealer.run(
                [sc_algorithm],
                subgraph,
                num_clusters=2,  # 强制拆分成两个社区
                whether_format_result=False,
            )
            new_communities = results[0].communities  # 拆分出的新社区

            # 计算新社区直径
            new_diameters = []
            new_community_dict = {}

            for new_comm in new_communities:
                subgraph = G.subgraph(new_comm)
                if nx.is_connected(subgraph):
                    new_diameter = nx.diameter(subgraph)
                else:
                    # 取最大连通分量的直径
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                    new_diameter = nx.diameter(G.subgraph(largest_cc))

                new_diameters.append(new_diameter)
                new_community_dict[next_community_id] = new_comm
                next_community_id += 1

            # **如果新的最大直径仍然等于原直径，则取消切割**
            if max(new_diameters) == largest_diameter:
                # print(
                #     f"❌ 取消拆分社区 {largest_community_id}，因为拆分后最大直径未降低 ({largest_diameter})"
                # )
                community_diameters.remove((largest_community_id, largest_diameter))
            else:
                # 删除原有社区
                del communities_dict[largest_community_id]
                community_diameters.remove((largest_community_id, largest_diameter))

                # 更新新社区信息
                for comm_id, new_comm in new_community_dict.items():
                    subgraph = G.subgraph(new_comm)

                    # 检查子图是否连通
                    # TODO 有可能不连通，所以设计一个更合理的算法逻辑
                    if nx.is_connected(subgraph):
                        new_diameter = nx.diameter(subgraph)
                    else:
                        # 取最大连通分量的直径
                        largest_cc = max(nx.connected_components(subgraph), key=len)
                        new_diameter = nx.diameter(G.subgraph(largest_cc))

                    # 存储社区信息
                    communities_dict[comm_id] = new_comm
                    community_diameters.append((comm_id, new_diameter))

        return communities_dict

    def merge_community(self, G, communities_dict, density_threshold):
        """
        根据紧密度的变化来合并社区
        Args:
            G: nx.graph
            communities_dict: A dictionary representing community detection results.
            density_threshold

        Returns:
            dict: A dictionary representing community detection results.
                Structure:
                    - Keys (int): Community IDs.
                    - Values (list of int): List of node IDs belonging to the corresponding community.
                Example:
                    {
                        0: [0, 1, 2, 3, 4],  # Community 0 contains nodes 0,1,2,3,4
                        1: [5, 6, 7, 8, 9],  # Community 1 contains nodes 5,6,7,8,9
                        2: [10, 11, 12]      # Community 2 contains nodes 10,11,12
                    }
        """

        # 社区之间尝试两两合并

        # 如果紧密度的偏差减小且大于阈值，则记录
        # 如何计算偏差？？？？
        # 选取偏差减小最多的两个社区进行合并

        # 循环，直到没有社区可以合并

        return communities_dict


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
    # number_of_point = 200  # 节点数
    # degree_exponent = 3  # 幂律指数
    # community_size_exponent = 3  # 社区大小幂律指数
    # average_degree = 6
    # min_degree = 2
    # min_community_size = 15
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
    pos = nx.spring_layout(G, seed=42)

    algorithmDealer = AlgorithmDealer()
    leiden_rare_algorithm = Leiden_Rare()
    results = algorithmDealer.run(
        [leiden_rare_algorithm], G, num_clusters=len(true_communities)
    )
    communities = results[0].communities
    # draw_communities(G, pos)
    # draw_communities(G, pos, communities)

    # 计算评估指标
    # 转化 truth_table 的格式
    truth_table = [
        [node, community_id]
        for community_id, nodes in enumerate(true_communities)
        for node in reversed(nodes)
    ]
    evaluation = CommunityDetectionMetrics(G, communities, truth_table)
    metrics = evaluation.evaluate()
    metrics["runtime"] = results[0].runtime

    # 可视化结果
    from algorithm.common.util.drawer import draw_communities

    draw_communities(G, pos, true_communities)
    draw_communities(G, pos, communities, title="Leiden_Rare", metrics=metrics)

    CommunityComparator(communities, true_communities).run()

# delete_G = G.subgraph(delete_communities).copy()
# leiden_algorithm = Leiden()
# sc_algorithm = SpectralCluster()
# delete_results = algorithmDealer.run(
#     [leiden_algorithm],
#     delete_G,
#     whether_format_result=False,
#     num_clusters=num_clusters-len(pre_communities)
# )
# delete_results = sorted(delete_results[0].communities, key=len, reverse=True)
# pos = nx.spring_layout(G, seed=42)
# draw_communities(G, pos, pre_communities)

# def get_pre_communities(self, num_clusters):
#     # 使用Leiden算法预处理
#     algorithmDealer = AlgorithmDealer()
#     leiden_algorithm = Leiden()
#     pre_results = algorithmDealer.run(
#         [leiden_algorithm], G, num_clusters=num_clusters
#     )
#     pre_communities = pre_results[0].communities
#
#     if num_clusters is None:
#         num_clusters = len(pre_communities)/2
#     num_clusters = num_clusters // 2
#
#     pre_communities = sorted(pre_communities, key=len, reverse=True)
#
#     delete_communities = pre_communities[num_clusters:]
#     delete_communities = sum(delete_communities, [])
#
#     pre_communities = pre_communities[:num_clusters]
#
#     return pre_communities, delete_communities
