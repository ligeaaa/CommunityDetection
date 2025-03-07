#!/usr/bin/env python
# coding=utf-8
import random

from algorithm.common.benchmark.benchmark_graph import create_graph

# 预定义不同类型的图的参数范围
graph_configs = {
    "type_A": {
        "number_of_point": (150, 200),
        "min_community_size": (10, 20),
        "degree_exponent": (2.0, 3.0),
        "community_size_exponent": (1.5, 2.5),
        "average_degree": (5, 15),
        "min_degree": (2, 5),
        "mixing_parameter": (0.1, 0.3),
    },
    "type_B": {
        "number_of_point": (500, 1000),
        "min_community_size": (30, 50),
        "degree_exponent": (2.5, 3.5),
        "community_size_exponent": (2.0, 3.0),
        "average_degree": (10, 20),
        "min_degree": (3, 7),
        "mixing_parameter": (0.2, 0.4),
    },
    # 可以继续添加更多类型
}


# 生成随机参数值
def random_params(param_range):
    if isinstance(param_range[0], int):
        return random.randint(*param_range)
    elif isinstance(param_range[0], float):
        return random.uniform(*param_range)
    return param_range[0]  # 适用于固定值


# 生成多个图
def generate_graphs(graph_type, num_graphs, random_seed=None):
    if graph_type not in graph_configs:
        raise ValueError(f"Graph type '{graph_type}' not found in configurations.")

    config = graph_configs[graph_type]  # 获取参数范围
    graphs = []

    for _ in range(num_graphs):
        params = {key: random_params(value) for key, value in config.items()}
        G, communities = create_graph(**params, seed=random_seed)  # 调用已有函数
        graphs.append((G, communities, params))  # 存储生成的图和参数信息

    return graphs


if __name__ == "__main__":
    # 示例：生成 3 个 type_A 类型的图
    generated_graphs = generate_graphs("type_A", 3, random_seed=42)

    # 查看生成的图的参数
    for i, (G, communities, params) in enumerate(generated_graphs):
        print(f"Graph {i+1}: {params}")
