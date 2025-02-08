#!/usr/bin/env python
# coding=utf-8
import unittest

import networkx as nx

from algorithm.algorithm_dealer import AlgorithmDealer, Algorithm
from algorithm.classic.louvain import Louvain
from algorithm.common.constant.test_data import test_raw_data, test_truth_table


class TestLouvain(unittest.TestCase):
    def testEasyGraph(self):
        # 示例输入：边的列表
        edge_list = test_raw_data
        truth_table = test_truth_table

        G = nx.Graph()
        G.add_edges_from(edge_list)
        algorithm_dealer = AlgorithmDealer()
        louvain_algorithm = Louvain()

        results = algorithm_dealer.run([louvain_algorithm], G)
        communities = results[0].communities

        self.assertCountEqual(communities, Algorithm.format_results(truth_table))
