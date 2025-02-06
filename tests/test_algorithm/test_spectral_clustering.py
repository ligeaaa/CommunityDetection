#!/usr/bin/env python
# coding=utf-8
import unittest

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.classic.spectral_clustering import SpectralCluster


class TestLouvain(unittest.TestCase):
    def testEasyGraph(self):
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

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        spectral_clustering_algorithm = SpectralCluster()

        results = algorithm_dealer.run(
            [spectral_clustering_algorithm], G, num_clusters=2
        )
        communities = results[0].communities

        self.assertCountEqual(communities, Algorithm.format_results(truth_table))
