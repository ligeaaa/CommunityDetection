#!/usr/bin/env python
# coding=utf-8
from typing import List

from networkx import Graph


class Algorithm:
    def __init__(self):
        self.algorithm_name = ...

    def process(self, G: Graph, **kwargs) -> list:
        ...


class AlgorithmResult:
    def __init__(self, algorithm_name: str, communities: list):
        self.algorithm_name = algorithm_name
        self.communities = communities

    def __repr__(self):
        return f"AlgorithmResult(name={self.algorithm_name}, communities={self.communities})"


class AlgorithmDealer:
    def __init__(self):
        self.results = []

    def process(self, algorithms: List[Algorithm], G: Graph, **kwargs):
        for algorithm in algorithms:
            communities = algorithm.process(G, **kwargs)
            result = AlgorithmResult(algorithm.algorithm_name, communities)
            self.results.append(result)
        return self.results
