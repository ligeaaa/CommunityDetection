#!/usr/bin/env python
# coding=utf-8
from typing import List

from networkx import Graph


class Algorithm:
    def __init__(self):
        self.algorithm_name = ...

    def run(self, G: Graph, **kwargs) -> list:
        raw_results = self.process(G, **kwargs)
        format_results = self.format_results(raw_results)
        return format_results

    def process(self, G: Graph, **kwargs) -> list: ...

    @staticmethod
    def format_results(raw_result: list) -> list:
        # Flatten all numbers and create sorted list
        all_numbers = sorted(set(num for sublist in raw_result for num in sublist))

        # Create a mapping from original numbers to consecutive indices
        number_mapping = {num: idx for idx, num in enumerate(all_numbers)}

        # Transform the original list using the mapping
        formatted_result = [
            [number_mapping[num] for num in sublist] for sublist in raw_result
        ]

        return formatted_result


class AlgorithmResult:
    def __init__(self, algorithm_name: str, communities: list):
        self.algorithm_name = algorithm_name
        self.communities = communities

    def __repr__(self):
        return f"AlgorithmResult(name={self.algorithm_name}, communities={self.communities})"


class AlgorithmDealer:
    def __init__(self):
        self.results = []

    def run(self, algorithms: List[Algorithm], G: Graph, **kwargs):
        for algorithm in algorithms:
            communities = algorithm.run(G, **kwargs)
            result = AlgorithmResult(algorithm.algorithm_name, communities)
            self.results.append(result)
        return self.results
