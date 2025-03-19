#!/usr/bin/env python
# coding=utf-8
import random
import time
from typing import List

from networkx import Graph

from algorithm.common.constant.constant_number import random_seed


class Algorithm:
    def __init__(self):
        self.random_seed = random_seed
        self.algorithm_name = ...
        self.G = ...
        self.version = "v0.01"
        random.seed(self.random_seed)

    def run(self, G: Graph, whether_format_result=True, **kwargs) -> list:
        raw_results = self.process(G, **kwargs)
        if whether_format_result:
            raw_results = self.format_results(raw_results)
        return raw_results

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
    def __init__(
        self, algorithm_name: str, communities: list, runtime: float, version: str
    ):
        self.algorithm_name = algorithm_name
        self.communities = communities
        self.runtime = runtime
        self.version = version

    def __repr__(self):
        return f"AlgorithmResult(name={self.algorithm_name}, communities={self.communities}, runtime={self.runtime:.4f}s)"


class AlgorithmDealer:
    def __init__(self):
        self.results = []

    def run(self, algorithms: List[Algorithm], G: Graph, **kwargs):
        for algorithm in algorithms:
            start_time = time.time()
            communities = algorithm.run(G.copy(), **kwargs)
            runtime = time.time() - start_time
            # 存储结果，包括运行时间
            result = AlgorithmResult(
                algorithm.algorithm_name, communities, runtime, algorithm.version
            )
            self.results.append(result)
        return self.results
