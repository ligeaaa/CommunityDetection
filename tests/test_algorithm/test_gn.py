#!/usr/bin/env python
# coding=utf-8
import unittest

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.classic.GN import GN
from algorithm.common.constant.test_data import test_raw_data, test_truth_table


class TestGN(unittest.TestCase):
    def testEasyGraph(self):
        # 示例输入：边的列表
        edge_list = test_raw_data
        truth_table = test_truth_table

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        GN_algorithm = GN()

        results = algorithm_dealer.run([GN_algorithm], G)
        communities = results[0].communities

        self.assertCountEqual(communities, Algorithm.format_results(truth_table))
