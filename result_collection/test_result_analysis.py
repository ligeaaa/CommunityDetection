import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithm.common.util.read_pkl import read_data

# 结果文件目录
result_dir = r"D:\code\FYP\CommunityDetection\result_collection\result\test"


def calculate_precision_recall(data_list):
    """计算每个数据集中不同社区的精确率和召回率，并按真实社区规模排序"""
    all_sorted_metrics = []  # 存储所有数据集的社区指标
    max_community_count = 0  # 记录数据集中最大的社区数量

    for data in data_list:
        community_stats = data["metrics"]["acc_per_community"]["community_stats"]
        community_metrics = []

        for community_id, stats in community_stats.items():
            TP, FP, FN = stats["TP"], stats["FP"], stats["FN"]
            real_size = TP + FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 计算精确率
            recall = TP / real_size if real_size > 0 else 0  # 计算召回率

            community_metrics.append((real_size, precision, recall))

        # 按真实社区规模（real_size）降序排序
        community_metrics.sort(reverse=True, key=lambda x: x[0])
        max_community_count = max(max_community_count, len(community_metrics))

        all_sorted_metrics.append(community_metrics)

    return all_sorted_metrics, max_community_count


def compute_average_metrics(all_sorted_metrics, max_community_count):
    """计算每个排名下的平均精确率和召回率，同时统计每个排名的社区数量"""
    aggregated_metrics = [[] for _ in range(max_community_count)]
    rank_counts = [0] * max_community_count  # 统计每个 rank 有多少个社区

    for dataset_metrics in all_sorted_metrics:
        for rank, (size, precision, recall) in enumerate(dataset_metrics):
            aggregated_metrics[rank].append((precision, recall))
            rank_counts[rank] += 1

    avg_results = []
    for rank, metrics in enumerate(aggregated_metrics):
        if metrics:
            avg_precision = np.mean([m[0] for m in metrics])
            avg_recall = np.mean([m[1] for m in metrics])
            avg_results.append((rank + 1, avg_precision, avg_recall, rank_counts[rank]))

    return avg_results


if __name__ == "__main__":
    data_list = read_data(
        result_dir,
        algorithm="Louvain",
        # point_level="level2",
        # density_level="level3",
        # community_size_level="level1"
    )

    # 计算精确率和召回率
    all_sorted_metrics, max_community_count = calculate_precision_recall(data_list)
    avg_results = compute_average_metrics(all_sorted_metrics, max_community_count)

    # 创建 DataFrame 并显示结果
    df = pd.DataFrame(
        avg_results, columns=["Rank", "Avg Precision", "Avg Recall", "Community Count"]
    )
    print(df.to_string(index=False))

    # 提取数据
    ranks = [row[0] for row in avg_results]
    avg_precision = [row[1] for row in avg_results]
    avg_recall = [row[2] for row in avg_results]
    community_counts = [row[3] for row in avg_results]

    # 绘制图表
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 画 Precision 和 Recall
    ax1.plot(
        ranks,
        avg_precision,
        marker="o",
        linestyle="-",
        label="Avg Precision",
        color="tab:blue",
    )
    ax1.plot(
        ranks,
        avg_recall,
        marker="s",
        linestyle="-",
        label="Avg Recall",
        color="tab:orange",
    )

    # 轴标签
    ax1.set_xlabel("Community Rank")
    ax1.set_ylabel("Score")
    ax1.set_title("Average Precision and Recall by Community Rank")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # 创建第二个 y 轴，显示社区数量
    ax2 = ax1.twinx()
    ax2.bar(ranks, community_counts, alpha=0.3, color="gray", label="Community Count")
    ax2.set_ylabel("Community Count")

    # 添加社区数量的标签
    for i, count in enumerate(community_counts):
        ax2.text(ranks[i], count + 0.2, str(count), ha="center", fontsize=10)

    ax2.legend(loc="upper right")

    # 显示图表
    plt.show()
