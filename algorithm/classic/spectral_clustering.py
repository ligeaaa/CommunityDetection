#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.util.CommunityCompare import CommunityComparator
from algorithm.common.util.drawer import draw_communities
from algorithm.common.util.result_evaluation import CommunityDetectionMetrics


class SpectralCluster(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "SpectralCluster"

    import numpy as np
    import networkx as nx
    from sklearn.cluster import KMeans
    from scipy.sparse.linalg import eigsh

    def process(self, G: nx.Graph, num_clusters=2, method="normalized_sym", **kwargs):
        """
        Perform spectral clustering on the given graph using different methods.

        Supported Methods:
        - "unnormalized": Unnormalized Spectral Clustering (Von Luxburg, 2007)
        - "normalized_rw": Normalized Spectral Clustering (Shi & Malik, 2000)
        - "normalized_sym": Normalized Spectral Clustering (Ng, Jordan & Weiss, 2002)

        Unnormalized Spectral Clustering:
        1. Construct the similarity graph and extract its weighted adjacency matrix W.
        2. Compute the degree matrix D.
        3. Compute the unnormalized graph Laplacian L = D - W.
        4. Compute the first k eigenvectors of L.
        5. Form a matrix U where columns correspond to the eigenvectors.
        6. Use k-means clustering on the rows of U to assign nodes into clusters.
        7. Return the final cluster assignments.

        The other 2 ways are similarly,
        In normalized_rw: L_(rw) := D^(−1)L
        In normalized_sym: L_(sym) := D^(-1/2)WD^(-1/2)

        Args:
            G (networkx.Graph): An undirected, weighted graph where nodes represent data points.
            num_clusters (int): Number of clusters (default: 2).
            method (str): The spectral clustering method to use. Default is "normalized_sym". Can be
                - "unnormalized"
                - "normalized_rw"
                - "normalized_sym"

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
            [1] Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and Computing, 17(4), 395-416.
                https://doi.org/10.1007/s11222-007-9033-z.
            [2] Shi, J. and Malik, J. (2000). Normalized cuts and image segmentation. IEEE Transactions on Pattern
                Analysis and Machine Intelligence, 22(8), 888 – 905.
            [3] Ng, A., Jordan, M., and Weiss, Y. (2002). On spectral clustering: analysis and an algorithm. In
                T. Dietterich, S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems
                14 (pp. 849 856). MIT Press.
        """

        # Step 1: Construct the weighted adjacency matrix W
        W = nx.to_numpy_array(G)

        # Step 2: Compute the degree matrix D
        D = np.diag(W.sum(axis=1))

        # Step 3: Compute the Laplacian matrix based on the selected method
        if method == "unnormalized":
            # Unnormalized Laplacian: L = D - W
            L = D - W
        elif method == "normalized_rw":
            # Shi and Malik's Normalized Laplacian: Solve Lu = λDu
            L = D - W
            D_inv = np.linalg.inv(D)  # Compute D⁻¹
            L = np.dot(D_inv, L)  # Solve generalized eigenproblem L u = λ D u
        elif method == "normalized_sym":
            # Ng, Jordan, and Weiss's Normalized Laplacian: L_sym = D^(-1/2) * L * D^(-1/2)
            L = D - W
            D_diag = np.diag(D)  # 取 D 的对角元素
            D_inv_sqrt_diag = np.zeros_like(D_diag)
            D_inv_sqrt_diag[D_diag > 0] = 1.0 / np.sqrt(
                D_diag[D_diag > 0]
            )  # 仅对非零度节点取逆
            D_inv_sqrt = np.diag(D_inv_sqrt_diag)  # 转换回对角矩阵
            L = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)  # L_sym
        else:
            raise ValueError(
                "Invalid method. Choose from ['unnormalized', 'shi_malik', 'ng_jordan_weiss']"
            )

        # Step 4: Compute the first k eigenvectors of L
        eigenvalues, eigenvectors = eigsh(L, k=num_clusters, which="SM")

        # Step 5: Form the matrix U with the eigenvectors as columns
        U = eigenvectors

        if method == "normalized_sym":
            # Normalize rows of U to have unit norm
            U = U / np.linalg.norm(U, axis=1, keepdims=True)

        # Step 6: Apply k-means clustering on the rows of U
        kmeans = KMeans(
            n_clusters=num_clusters, n_init=10, random_state=self.random_seed
        )
        labels = kmeans.fit_predict(U)

        # Step 7: Assign nodes to clusters
        best_communities = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(labels):
            best_communities[label].append(list(G.nodes())[idx])

        return best_communities


if __name__ == "__main__":
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
    pos = nx.spring_layout(G, seed=42)
    draw_communities(G, pos, true_communities)
    algorithmDealer = AlgorithmDealer()
    sc_algorithm = SpectralCluster()
    results = algorithmDealer.run([sc_algorithm], G, num_clusters=len(true_communities))
    communities = results[0].communities
    pos = nx.spring_layout(G, seed=42)
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

    draw_communities(G, pos, communities, title="Leiden", metrics=metrics)

    CommunityComparator(communities, true_communities).run()
