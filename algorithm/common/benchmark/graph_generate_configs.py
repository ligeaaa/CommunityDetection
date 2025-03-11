import os
import pickle
import random
import itertools
import time

import networkx as nx
from matplotlib import pyplot as plt

from algorithm.common.benchmark.benchmark_graph import create_graph
from algorithm.common.constant.constant_number import random_seed
from algorithm.common.util.drawer import draw_communities

# 预定义不同类型的图的参数范围
node_sizes = {"level1": (50, 100), "level2": (400, 500), "level3": (2000, 3000)}
density_levels = {"level1": (2, 3), "level2": (4, 5), "level3": (6, 7)}
# 表明最小社区size占总点数的百分比
community_counts = {"level1": (0.01, 0.05), "level2": (0.1, 0.2), "level3": (0.4, 0.5)}


def random_params(param_range):
    """根据范围随机生成参数值"""
    if isinstance(param_range[0], int):
        return random.randint(*param_range)
    elif isinstance(param_range[0], float):
        return random.uniform(*param_range)
    return param_range[0]


def generate_graphs(
    node_category,
    density_category,
    community_category,
    num_graphs,
    title,
    random_seed=None,
):
    """生成指定类别的多个基准测试图"""
    if (
        node_category not in node_sizes
        or density_category not in density_levels
        or community_category not in community_counts
    ):
        raise ValueError("无效的图类别")

    graphs = []
    for _ in range(num_graphs):

        min_degree = random_params(density_levels[density_category])
        average_degree = random.randint(int(min_degree * 1.5), int(min_degree * 2))

        config = {
            "number_of_point": node_sizes[node_category],
            "average_degree": average_degree,
            "min_community_size": community_counts[community_category],
            "degree_exponent": (2.0, 3.5),  # 保持默认范围
            "community_size_exponent": (1.5, 3.0),
            "min_degree": min_degree,
            "mixing_parameter": (0.05, 0.2),
        }

        params = {
            key: random_params(value) if isinstance(value, tuple) else value
            for key, value in config.items()
        }
        params["min_community_size"] = int(
            params["number_of_point"] * params["min_community_size"]
        )
        G, communities = create_graph(**params, seed=random_seed)
        graphs.append((G, communities, params, title))

    return graphs


if __name__ == "__main__":
    output_dir = "generated_graphs"
    # 生成所有27种类型，每种类型生成1个图
    categories = ["level1", "level2", "level3"]
    all_graphs = []
    # random.seed(random_seed)

    for node_cat, density_cat, community_cat in itertools.product(categories, repeat=3):
        title = (
            f"point-{node_cat}, density-{density_cat}, community_size-{community_cat}"
        )
        print(title)
        retry_count = 0
        while retry_count < 10:
            try:
                generated_graphs = generate_graphs(
                    node_cat,
                    density_cat,
                    community_cat,
                    10,
                    title,
                    # random_seed=random_seed,
                )
                all_graphs.extend(generated_graphs)
                break  # 成功生成则跳出循环
            except ValueError as e:
                print(f"错误: {e}, 正在重试 ({retry_count + 1}/10)")
                retry_count += 1
        else:
            print(
                f"连续10次失败，跳过类型: 点数-{node_cat}, 密度-{density_cat}, 社区size-{community_cat}"
            )

    # 保存数据与图像
    for i, (G, communities, params, title) in enumerate(all_graphs):
        time.sleep(1)
        print(f"Graph {i + 1}: {params}")
        pos = nx.spring_layout(G, seed=random_seed)
        fig = draw_communities(G, pos, communities, title=title, metrics=params)

        # 规范化命名
        filename_prefix = f"{i}_{title}"

        # 保存图像
        fig_path = os.path.join(output_dir, f"{filename_prefix}.png")
        fig.savefig(fig_path)
        plt.close(fig)

        # 保存数据（参数和社区信息）
        data_path = os.path.join(output_dir, f"{filename_prefix}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(
                {
                    "graph": G,
                    "communities": communities,
                    "params": params,
                    "title": title,
                },
                f,
            )

        print(f"已保存: {fig_path} 和 {data_path}")
