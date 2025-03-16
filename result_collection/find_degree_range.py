import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# 结果文件目录
pkl_directory = r"D:\code\FYP\CommunityDetection\algorithm\common\benchmark\generated_graphs"  # pkl文件读取目录

if __name__ == "__main__":
    # 获取目录下所有的 .pkl 文件
    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

    # 读取所有 pkl 文件
    data_list = []
    for file in pkl_files:
        file_path = os.path.join(pkl_directory, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data["title"] = os.path.splitext(file)[0]
            data_list.append(data)

    # 解析数据，筛选社区数量大于3的数据集
    filtered_data = [data for data in data_list if len(data["communities"]) > 3]

    degree_percentiles = []
    colors = []
    dataset_indices = []
    dataset_index = 0

    for data in filtered_data:
        G = data["graph"]

        # 计算全局度数分布
        all_degrees = [G.degree(n) for n in G.nodes()]
        all_degrees.sort()

        # 计算社区大小，并按照大小排序
        community_sizes = [
            (len(community), community) for community in data["communities"]
        ]
        community_sizes.sort(reverse=True, key=lambda x: x[0])

        # 选取前33%、后33%和中间33%的社区
        num_communities = len(community_sizes)
        top_33_count = max(1, num_communities // 3)
        bottom_33_count = max(1, num_communities // 3)

        top_33_communities = community_sizes[:top_33_count]
        bottom_33_communities = community_sizes[-bottom_33_count:]
        middle_33_communities = community_sizes[top_33_count:-bottom_33_count]

        # 计算度数的百分位数位置，并为每个社区的最大度数节点和其余节点赋予不同的颜色
        for size, community in top_33_communities:
            max_degree_node = max(community, key=lambda n: G.degree(n))
            for node in community:
                degree = G.degree(node)
                percentile = (
                    np.searchsorted(all_degrees, degree, side="right")
                    / len(all_degrees)
                ) * 100
                dataset_indices.append(dataset_index)
                if node == max_degree_node:
                    degree_percentiles.append(percentile)
                    colors.append("darkblue")
                else:
                    degree_percentiles.append(percentile)
                    colors.append("lightblue")

        for size, community in bottom_33_communities:
            max_degree_node = max(community, key=lambda n: G.degree(n))
            for node in community:
                degree = G.degree(node)
                percentile = (
                    np.searchsorted(all_degrees, degree, side="right")
                    / len(all_degrees)
                ) * 100
                dataset_indices.append(dataset_index)
                if node == max_degree_node:
                    degree_percentiles.append(percentile)
                    colors.append("darkred")
                else:
                    degree_percentiles.append(percentile)
                    colors.append("lightcoral")

        for size, community in middle_33_communities:
            for node in community:
                degree = G.degree(node)
                percentile = (
                    np.searchsorted(all_degrees, degree, side="right")
                    / len(all_degrees)
                ) * 100
                dataset_indices.append(dataset_index)
                degree_percentiles.append(percentile)
                colors.append("gray")

        dataset_index += 1

    # 画点图，按照优先级排序：深红色=深蓝色 > 浅红色=浅蓝色 > 灰色
    sorted_indices = sorted(
        range(len(colors)),
        key=lambda i: (
            colors[i] in ["darkred", "darkblue"],
            colors[i] in ["lightcoral", "lightblue"],
        ),
    )
    sorted_x = [dataset_indices[i] for i in sorted_indices]
    sorted_y = [degree_percentiles[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.scatter(sorted_x, sorted_y, color=sorted_colors, alpha=0.8)

    plt.xlabel("Dataset Index")
    plt.ylabel("Degree Percentile (%)")
    plt.title(
        "Degree Percentile Distribution in Top, Middle, and Bottom 33% Communities"
    )
    plt.grid(True)

    plt.show()
