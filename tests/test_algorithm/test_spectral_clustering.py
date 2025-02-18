#!/usr/bin/env python
# coding=utf-8
import unittest

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.classic.spectral_clustering import SpectralCluster
from algorithm.common.constant.test_data import test_truth_table, test_raw_data


class TestSC(unittest.TestCase):
    def testEasyGraph_unnormalized(self):
        # 示例输入：边的列表
        edge_list = test_raw_data
        truth_table = test_truth_table

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        spectral_clustering_algorithm = SpectralCluster()

        results = algorithm_dealer.run(
            [spectral_clustering_algorithm], G, num_clusters=2, method="unnormalized"
        )
        communities = results[0].communities

        self.assertCountEqual(communities, Algorithm.format_results(truth_table))

    def testEasyGraph_normalized_rw(self):
        # 示例输入：边的列表
        edge_list = test_raw_data
        truth_table = test_truth_table

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        spectral_clustering_algorithm = SpectralCluster()

        results = algorithm_dealer.run(
            [spectral_clustering_algorithm], G, num_clusters=2, method="normalized_rw"
        )
        communities = results[0].communities

        self.assertCountEqual(communities, Algorithm.format_results(truth_table))

    def testEasyGraph_normalized_sym(self):
        # 示例输入：边的列表
        edge_list = test_raw_data
        truth_table = test_truth_table

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        spectral_clustering_algorithm = SpectralCluster()

        results = algorithm_dealer.run(
            [spectral_clustering_algorithm], G, num_clusters=2, method="normalized_sym"
        )
        communities = results[0].communities

        self.assertCountEqual(communities, Algorithm.format_results(truth_table))
