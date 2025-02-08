#!/usr/bin/env python
# coding=utf-8
import unittest

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.classic.SBM import SBM
from algorithm.classic.louvain import Louvain
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.constant.test_data import test_raw_data, test_truth_table


class TestAlgorithmDealer(unittest.TestCase):

    def testFormatResults(self):
        raw_results = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        format_results = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.assertCountEqual(Algorithm.format_results(raw_results), format_results)

    def testMulAlgorithm(self):
        # 示例输入：边的列表
        edge_list = test_raw_data
        truth_table = test_truth_table

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        louvain_algorithm = Louvain()
        sbm_algorithm = SBM()
        spectral_clustering_algorithm = SpectralCluster()

        results = algorithm_dealer.run(
            [louvain_algorithm, sbm_algorithm, spectral_clustering_algorithm],
            G,
            num_clusters=2,
        )

        for result in results:
            communities = result.communities
            self.assertCountEqual(communities, Algorithm.format_results(truth_table))
