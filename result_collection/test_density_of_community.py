#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# **参数**
pkl_directory = r"D:\code\FYP\CommunityDetection\algorithm\common\benchmark\generated_graphs"  # `pkl` 文件读取目录
result_dir = "result/test"  # `pkl` 数据保存目录
os.makedirs(result_dir, exist_ok=True)  # **确保目录存在**

# **存储 `社区 size` 对应的 `density`**
size_density_map = defaultdict(list)

# **主程序**
if __name__ == "__main__":

    # **获取目录下所有的 .pkl 文件**
    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

    # **读取所有 pkl 文件**
    data_list = []
    for file in pkl_files:
        file_path = os.path.join(pkl_directory, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data["title"] = os.path.splitext(file)[0]
            data_list.append(data)

    # 初始化存储 size-density 数据的字典
    size_density_map = defaultdict(list)

    for data in data_list:
        G = data["graph"]
        num_nodes = G.number_of_nodes()

        # **获取社区划分**
        communities = data[
            "communities"
        ]  # 这里假设 `communities` 是 `List[List[node]]` 格式
        title = data["title"]

        print(f"\n🔹 处理数据集: {title} (总节点数: {num_nodes})")

        # **计算每个社区的密度**
        for community in communities:
            size = len(community)
            density = nx.density(G.subgraph(community))
            size_density_map[size].append(density)

    # **计算每个 `size` 的 `平均 density`，然后计算 `size * avg_density`**
    size_density_product = {
        size: size * np.mean(densities) for size, densities in size_density_map.items()
    }

    # **分箱处理（Bin sizes into intervals of 10）**
    bin_width = 10  # 每 10 个单位为一组
    bins = defaultdict(list)

    for size, density_product in size_density_product.items():
        bin_key = (size // bin_width) * bin_width  # 计算所属的 bin
        bins[bin_key].append(density_product)

    # **计算每个 bin 的平均值**
    bin_means = {bin_key: np.mean(values) for bin_key, values in bins.items()}

    # **绘制 Bin 处理后的 `size-density product` 柱状图**
    fig, ax = plt.subplots(figsize=(8, 6))
    bin_centers = list(bin_means.keys())
    bin_values = list(bin_means.values())

    ax.bar(
        bin_centers,
        bin_values,
        width=bin_width * 0.8,
        color="blue",
        alpha=0.7,
        edgecolor="black",
        label="Binned Average",
    )

    # **设置图表属性**
    ax.set_xlabel("Community Size (Binned)", fontsize=14)
    ax.set_ylabel("Size × Average Density", fontsize=14)
    ax.set_title("Binned Community Size vs Size × Average Density", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()

    # **保存图像**
    plt.savefig(
        os.path.join(result_dir, "binned_size_density_product_plot.png"), dpi=300
    )
    plt.show()

    print("\n✅ 已完成 `size-density product` 关系统计并生成柱状图！")
