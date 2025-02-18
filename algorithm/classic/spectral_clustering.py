#!/usr/bin/env python
# coding=utf-8
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from algorithm.algorithm_dealer import Algorithm, AlgorithmDealer
from algorithm.common.constant.test_data import test_raw_data, test_truth_table
from algorithm.common.util.drawer import draw_communities


class SpectralCluster(Algorithm):
    def __init__(self):
        super().__init__()
        self.algorithm_name = "SpectralCluster"

    def process(self, G: nx.Graph, num_clusters=2, **kwargs):
        """
        Perform unnormalized spectral clustering on the given graph.

        The algorithm follows these main steps:

        1. Construct the similarity graph and extract its weighted adjacency matrix W.
        2. Compute the degree matrix D.
        3. Compute the unnormalized graph Laplacian L = D - W.
        4. Compute the first k eigenvectors of L.
        5. Form a matrix U where columns correspond to the eigenvectors.
        6. Use k-means clustering on the rows of U to assign nodes into clusters.
        7. Return the final cluster assignments.

        Args:
            G (networkx.Graph): An undirected, weighted graph where nodes represent data points
                                and edge weights represent pairwise similar
            num_clusters (int): Number of clusters (default: 2).

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
            [1] Von Luxburg, U. (2007) ‘023_A tutorial on spectral clustering’, Statistics and Computing,
            17(4), pp. 395–416. Available at: https://doi.org/10.1007/s11222-007-9033-z.
        """
        # Step 1: Construct the weighted adjacency matrix W
        W = nx.to_numpy_array(G)

        # Step 2: Compute the degree matrix D
        D = np.diag(W.sum(axis=1))

        # Step 3: Compute the unnormalized Laplacian L = D - W
        L = D - W

        # Step 4: Compute the first k eigenvectors of L
        eigenvalues, eigenvectors = eigsh(L, k=num_clusters, which="SM")

        # Step 5: Form the matrix U with the eigenvectors as columns
        U = eigenvectors

        # Step 6: Apply k-means clustering on the rows of U
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(U)

        # Step 7: Assign nodes to clusters
        best_communities = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(labels):
            best_communities[label].append(list(G.nodes())[idx])

        return best_communities


if __name__ == "__main__":
    edge_list = test_raw_data
    truth_table = test_truth_table
    G = nx.Graph()
    G.add_edges_from(edge_list)
    algorithmDealer = AlgorithmDealer()
    SC_algorithm = SpectralCluster()
    results = algorithmDealer.run([SC_algorithm], G, num_clusters=2)
    communities = results[0].communities
    pos = nx.spring_layout(G)
    draw_communities(G, pos)
    draw_communities(G, pos, communities)
