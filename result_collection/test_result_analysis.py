import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from algorithm.common.util.read_pkl import read_data


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


def compute_avg_metrics(data_list):
    """
    计算数据集的平均运行时间、平均精确率、平均召回率、平均模块度
    """
    runtimes = []
    precisions = []
    recalls = []
    modularities = []

    for data in data_list:
        metrics = data["metrics"]
        runtimes.append(metrics["runtime"])
        precisions.append(metrics["Precision"])
        recalls.append(metrics["Recall"])
        modularities.append(metrics["Modularity"])

    avg_runtime = np.mean(runtimes) if runtimes else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_modularity = np.mean(modularities) if modularities else 0

    return avg_runtime, avg_precision, avg_recall, avg_modularity


def show_comparison_precision_recall(
    data_list_1, data_list_2, label_1="Dataset 1", label_2="Dataset 2"
):
    """
    对比两组数据集的 Precision 和 Recall，并在同一张图上展示
    """

    # 计算两组数据的指标
    all_sorted_metrics_1, max_community_count_1 = calculate_precision_recall(
        data_list_1
    )
    avg_results_1 = compute_average_metrics(all_sorted_metrics_1, max_community_count_1)

    all_sorted_metrics_2, max_community_count_2 = calculate_precision_recall(
        data_list_2
    )
    avg_results_2 = compute_average_metrics(all_sorted_metrics_2, max_community_count_2)

    # 创建 DataFrame 并显示结果
    df_1 = pd.DataFrame(
        avg_results_1,
        columns=["Rank", "Avg Precision", "Avg Recall", "Community Count"],
    )
    df_2 = pd.DataFrame(
        avg_results_2,
        columns=["Rank", "Avg Precision", "Avg Recall", "Community Count"],
    )

    print(f"\n🔹 {label_1} Results:")
    print(df_1.to_string(index=False))
    print(f"\n🔹 {label_2} Results:")
    print(df_2.to_string(index=False))

    # 提取数据
    ranks_1 = [row[0] for row in avg_results_1]
    avg_precision_1 = [row[1] for row in avg_results_1]
    avg_recall_1 = [row[2] for row in avg_results_1]
    community_counts_1 = [row[3] for row in avg_results_1]

    ranks_2 = [row[0] for row in avg_results_2]
    avg_precision_2 = [row[1] for row in avg_results_2]
    avg_recall_2 = [row[2] for row in avg_results_2]
    community_counts_2 = [row[3] for row in avg_results_2]

    # 绘制图表
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 画 Precision 和 Recall（数据集1）
    ax1.plot(
        ranks_1,
        avg_precision_1,
        marker="o",
        linestyle="-",
        label=f"{label_1} - Avg Precision",
        color="tab:blue",
    )
    ax1.plot(
        ranks_1,
        avg_recall_1,
        marker="s",
        linestyle="-",
        label=f"{label_1} - Avg Recall",
        color="tab:orange",
    )

    # 画 Precision 和 Recall（数据集2）
    ax1.plot(
        ranks_2,
        avg_precision_2,
        marker="^",
        linestyle="--",
        label=f"{label_2} - Avg Precision",
        color="tab:green",
    )
    ax1.plot(
        ranks_2,
        avg_recall_2,
        marker="D",
        linestyle="--",
        label=f"{label_2} - Avg Recall",
        color="tab:red",
    )

    # 轴标签
    ax1.set_xlabel("Community Rank")
    ax1.set_ylabel("Score")
    ax1.set_title("Comparison of Average Precision and Recall by Community Rank")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # 创建第二个 y 轴，显示社区数量
    ax2 = ax1.twinx()
    ax2.bar(
        ranks_1,
        community_counts_1,
        alpha=0.3,
        color="gray",
        label=f"{label_1} - Community Count",
    )
    ax2.bar(
        ranks_2,
        community_counts_2,
        alpha=0.3,
        color="black",
        label=f"{label_2} - Community Count",
    )

    ax2.set_ylabel("Community Count")

    # 添加社区数量的标签
    for i, count in enumerate(community_counts_1):
        ax2.text(ranks_1[i], count + 0.2, str(count), ha="center", fontsize=10)

    for i, count in enumerate(community_counts_2):
        ax2.text(ranks_2[i], count + 0.2, str(count), ha="center", fontsize=10)

    ax2.legend(loc="upper right")

    # 显示图表
    plt.show()


def show_comparison_avg_metrics(
    data_list_1, data_list_2, label_1="Dataset 1", label_2="Dataset 2"
):
    """
    对比两组数据的 平均运行时间、准确率、模块度，并在同一张图上展示
    """

    avg_runtime_1, avg_precision_1, avg_recall_1, avg_modularity_1 = (
        compute_avg_metrics(data_list_1)
    )
    avg_runtime_2, avg_precision_2, avg_recall_2, avg_modularity_2 = (
        compute_avg_metrics(data_list_2)
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Runtime", "Precision", "Recall", "Modularity"],
            label_1: [avg_runtime_1, avg_precision_1, avg_recall_1, avg_modularity_1],
            label_2: [avg_runtime_2, avg_precision_2, avg_recall_2, avg_modularity_2],
        }
    )

    print(metrics_df.to_string(index=False))

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.3
    x_labels = ["Runtime", "Precision", "Recall", "Modularity"]
    x_indexes = np.arange(len(x_labels))

    ax.bar(
        x_indexes - bar_width / 2,
        [avg_runtime_1, avg_precision_1, avg_recall_1, avg_modularity_1],
        width=bar_width,
        label=label_1,
        color="tab:blue",
        alpha=0.7,
    )

    ax.bar(
        x_indexes + bar_width / 2,
        [avg_runtime_2, avg_precision_2, avg_recall_2, avg_modularity_2],
        width=bar_width,
        label=label_2,
        color="tab:orange",
        alpha=0.7,
    )

    # 添加标签
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Value")
    ax.set_title("Comparison of Average Runtime, Accuracy, and Modularity")
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # 显示图表
    plt.show()


def compare_all_conditions(
    result_dir,
    algorithm_1,
    algorithm_2,
    max_rank_number=10,
    sub_fig_width=5,
    sub_fig_height=4,
    hspace=0.5,
    wspace=0.5,
):
    """
    对比所有 (3 x 3 x 3) = 27 种不同参数组合的 Precision-Recall 和 Avg Metrics，
    并将所有小图拼成一张大图。

    :param result_dir: 数据目录
    :param algorithm_1: 算法1的名称
    :param algorithm_2: 算法2的名称
    :param max_rank_number: 统一 rank 长度（默认为 10）
    :param sub_fig_width: 单个子图的宽度
    :param sub_fig_height: 单个子图的高度
    :param hspace: 子图之间的纵向间距
    :param wspace: 子图之间的横向间距
    """

    def plot_precision_recall(
        data_list_1, data_list_2, ax, label, algo_1, algo_2, max_rank_number=10
    ):
        """
        计算 Precision-Recall，并在当前子图 `ax` 中绘制
        同时绘制 rank 对应的 `社区数量`（仅绘制一根柱状图）
        """

        # 计算 Precision & Recall
        all_sorted_metrics_1, max_comm_count_1 = calculate_precision_recall(data_list_1)
        avg_results_1 = compute_average_metrics(all_sorted_metrics_1, max_comm_count_1)

        all_sorted_metrics_2, max_comm_count_2 = calculate_precision_recall(data_list_2)
        avg_results_2 = compute_average_metrics(all_sorted_metrics_2, max_comm_count_2)

        # 固定 x 轴长度为 max_rank_number
        ranks = np.arange(1, max_rank_number + 1)

        def pad_or_truncate(arr, length, fill_value=0):
            """如果数据长度大于 `length`，截断；如果不足 `length`，补全 `fill_value`"""
            arr = arr[:length]  # 截断
            return np.pad(
                arr, (0, max(0, length - len(arr))), constant_values=fill_value
            )  # 补全 `fill_value`

        avg_precision_1 = pad_or_truncate(
            [row[1] for row in avg_results_1], max_rank_number
        )
        avg_recall_1 = pad_or_truncate(
            [row[2] for row in avg_results_1], max_rank_number
        )
        avg_precision_2 = pad_or_truncate(
            [row[1] for row in avg_results_2], max_rank_number
        )
        avg_recall_2 = pad_or_truncate(
            [row[2] for row in avg_results_2], max_rank_number
        )

        # **计算每个 rank 对应的社区数量**（只绘制一根柱状图）
        community_counts = pad_or_truncate(
            [row[3] for row in avg_results_1], max_rank_number
        )

        # **创建第二个 y 轴** 用于显示 `社区数量`
        ax2 = ax.twinx()

        # **绘制 Precision 和 Recall 曲线**
        ax.plot(
            ranks,
            avg_precision_1,
            linestyle="-",
            label=f"{algo_1} Precision",
            color="tab:blue",
        )
        ax.plot(
            ranks,
            avg_recall_1,
            linestyle="--",
            label=f"{algo_1} Recall",
            color="tab:blue",
        )
        ax.plot(
            ranks,
            avg_precision_2,
            linestyle="-",
            label=f"{algo_2} Precision",
            color="tab:red",
        )
        ax.plot(
            ranks,
            avg_recall_2,
            linestyle="--",
            label=f"{algo_2} Recall",
            color="tab:red",
        )

        # **绘制社区数量的柱状图（只有一根）**
        ax2.bar(
            ranks,
            community_counts,
            width=0.4,
            alpha=0.4,
            color="gray",
            label="Community Count",
        )

        # **设置坐标轴信息**
        ax.set_xlabel("Community Rank", fontsize=12)
        ax.set_ylabel("Precision / Recall", fontsize=12)
        ax2.set_ylabel("Community Count", fontsize=12)

        ax.set_xticks(ranks)
        ax2.set_xticks(ranks)

        # **在柱状图上显示 `社区数量`**
        for i, count in enumerate(community_counts):
            ax2.text(ranks[i], count + 0.2, str(count), ha="center", fontsize=10)

        ax.set_title(label, fontsize=14)

    def plot_avg_metrics(data_list_1, data_list_2, ax, label, algo_1, algo_2):
        """
        计算平均 Metrics 并绘制到当前子图 `ax`
        - `Runtime` 使用 **左 Y 轴**
        - `Precision, Recall, Modularity` 使用 **右 Y 轴**，范围固定为 `0-1`
        """

        def compute_avg_metrics(data_list):
            runtimes, precisions, recalls, modularities = [], [], [], []
            for data in data_list:
                metrics = data["metrics"]
                runtimes.append(metrics["runtime"])
                precisions.append(metrics["Precision"])
                recalls.append(metrics["Recall"])
                modularities.append(metrics["Modularity"])

            return (
                np.mean(runtimes),
                np.mean(precisions),
                np.mean(recalls),
                np.mean(modularities),
            )

        avg_runtime_1, avg_precision_1, avg_recall_1, avg_modularity_1 = (
            compute_avg_metrics(data_list_1)
        )
        avg_runtime_2, avg_precision_2, avg_recall_2, avg_modularity_2 = (
            compute_avg_metrics(data_list_2)
        )

        # **X 轴标签**
        x_labels_1 = ["Runtime"]
        x_labels_2 = ["Precision", "Recall", "Modularity"]

        x_indexes_1 = np.array([0])  # Runtime 在 X 轴的位置
        x_indexes_2 = np.array([1, 2, 3])  # 其它 Metrics 在 X 轴的位置

        # **创建第二个 Y 轴**（右侧）
        ax2 = ax.twinx()

        # **绘制 Runtime（左 Y 轴）**
        ax.bar(
            x_indexes_1 - 0.2,
            [avg_runtime_1],
            width=0.4,
            label=f"{algo_1} Runtime",
            alpha=0.8,
            color="tab:blue",
        )
        ax.bar(
            x_indexes_1 + 0.2,
            [avg_runtime_2],
            width=0.4,
            label=f"{algo_2} Runtime",
            alpha=0.8,
            color="tab:orange",
        )

        # **绘制 Precision, Recall, Modularity（右 Y 轴）**
        ax2.bar(
            x_indexes_2 - 0.2,
            [avg_precision_1, avg_recall_1, avg_modularity_1],
            width=0.4,
            label=f"{algo_1} Precision/Recall/Modularity",
            alpha=0.7,
            color="tab:green",
        )

        ax2.bar(
            x_indexes_2 + 0.2,
            [avg_precision_2, avg_recall_2, avg_modularity_2],
            width=0.4,
            label=f"{algo_2} Precision/Recall/Modularity",
            alpha=0.7,
            color="tab:red",
        )

        # **设置 X 轴标签**
        ax.set_xticks(np.concatenate((x_indexes_1, x_indexes_2)))
        ax.set_xticklabels(x_labels_1 + x_labels_2, fontsize=12)

        # **设置 Y 轴范围**
        ax2.set_ylim(0, 1)  # **右 Y 轴范围固定在 0-1**
        ax.set_ylim(0, max(avg_runtime_1, avg_runtime_2) * 1.2)  # **左 Y 轴自动调整**

        # **设置 Y 轴标签**
        ax.set_ylabel("Runtime (seconds)", fontsize=12, color="tab:blue")
        ax2.set_ylabel(
            "Precision / Recall / Modularity", fontsize=12, color="tab:green"
        )

        ax.set_title(label, fontsize=14)

        # **在柱状图上添加数值**
        ax.text(
            x_indexes_1[0] - 0.1,
            avg_runtime_1 + 0.01,
            f"{avg_runtime_1:.3f}",
            ha="center",
            fontsize=10,
            color="tab:blue",
        )
        ax.text(
            x_indexes_1[0] + 0.1,
            avg_runtime_2 + 0.01,
            f"{avg_runtime_2:.3f}",
            ha="center",
            fontsize=10,
            color="tab:orange",
        )

        for i, v in enumerate([avg_precision_1, avg_recall_1, avg_modularity_1]):
            ax2.text(
                x_indexes_2[i] - 0.1,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                fontsize=10,
                color="tab:green",
            )

        for i, v in enumerate([avg_precision_2, avg_recall_2, avg_modularity_2]):
            ax2.text(
                x_indexes_2[i] + 0.1,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                fontsize=10,
                color="tab:red",
            )

        # **优化 X 轴间距**
        ax.margins(x=0.1)

        # **返回 legend 信息**
        handles1, labels1 = ax.get_legend_handles_labels()  # 左 Y 轴 legend
        handles2, labels2 = ax2.get_legend_handles_labels()  # 右 Y 轴 legend
        return handles1 + handles2, labels1 + labels2  # **合并 legend**

    # 可能的参数取值
    point_levels = ["level1", "level2", "level3"]
    density_levels = ["level1", "level2", "level3"]
    community_size_levels = ["level1", "level2", "level3"]

    # **动态计算大图尺寸**
    fig_width = 9 * sub_fig_width + (9 - 1) * wspace
    fig_height = 3 * sub_fig_height + (3 - 1) * hspace

    fig1, axes1 = plt.subplots(
        3, 9, figsize=(fig_width, fig_height), constrained_layout=True
    )  # Precision-Recall
    fig2, axes2 = plt.subplots(
        3, 9, figsize=(fig_width, fig_height), constrained_layout=True
    )  # Avg Metrics

    plot_index = 0  # 记录子图索引

    for p in point_levels:
        for d in density_levels:
            for c in community_size_levels:
                label = f"point-{p}, density-{d}, community-{c}"

                # 读取两组数据
                data_1 = read_data(
                    result_dir,
                    algorithm=algorithm_1,
                    point_level=p,
                    density_level=d,
                    community_size_level=c,
                )
                data_2 = read_data(
                    result_dir,
                    algorithm=algorithm_2,
                    point_level=p,
                    density_level=d,
                    community_size_level=c,
                )

                if not data_1 or not data_2:
                    print(f"⚠️ 跳过 {label}（数据缺失）")
                    continue

                print(f"🔍 处理 {label}...")

                # 获取当前子图
                row, col = divmod(plot_index, 9)  # 3行9列的索引
                ax1 = axes1[row, col]  # Precision-Recall 子图
                ax2 = axes2[row, col]  # Avg Metrics 子图

                # 计算 Precision-Recall 并绘制曲线
                plot_precision_recall(
                    data_1,
                    data_2,
                    ax1,
                    label,
                    algorithm_1,
                    algorithm_2,
                    max_rank_number,
                )

                # 计算 Avg Metrics 并绘制柱状图
                plot_avg_metrics(data_1, data_2, ax2, label, algorithm_1, algorithm_2)

                # 设置子图坐标轴标签
                ax1.set_xlabel("Rank", fontsize=12)
                ax1.set_ylabel("Score", fontsize=12)

                ax2.set_xlabel("Metric", fontsize=12)
                ax2.set_ylabel("Value", fontsize=12)

                plot_index += 1  # 递增索引

    # **设置大图标题**
    fig1.suptitle(
        "Precision-Recall Comparison Across 27 Conditions",
        fontsize=24,
        fontweight="bold",
    )
    fig2.suptitle(
        "Average Metrics Comparison Across 27 Conditions",
        fontsize=24,
        fontweight="bold",
    )

    # **仅在大图级别添加图例**
    handles1, labels1 = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc="upper right", fontsize=16)

    # **修复 `fig2` 缺少 `legend`**
    handles2, labels2 = [], []
    for i in range(3):
        for j in range(9):
            legend_handles, legend_labels = axes2[
                i, j
            ].get_legend_handles_labels()  # **获取 `legend`**
            handles2.extend(legend_handles)
            labels2.extend(legend_labels)

    # **去重，确保 `metrics` 图的图例完整**
    unique_legend = dict(zip(labels2, handles2))
    fig2.legend(
        unique_legend.values(), unique_legend.keys(), loc="upper right", fontsize=16
    )

    # **保存图像**
    fig1.savefig(
        r"result\precision_recall_comparison_grid.png", dpi=400, bbox_inches="tight"
    )
    fig2.savefig(
        r"result\avg_metrics_comparison_grid.png", dpi=400, bbox_inches="tight"
    )

    # **显示图表**
    plt.show()


if __name__ == "__main__":
    # 结果文件目录
    result_dir = r"D:\code\FYP\CommunityDetection\result_collection\result\test"
    # 是否比较所有组合（27种）
    compare_all_combinations = True
    # 对比的两种算法
    algorithm_name_1 = "Leiden_Rarev0.01"
    algorithm_name_2 = "Leiden"
    data_list_1 = read_data(
        result_dir,
        algorithm=algorithm_name_1,
    )
    data_list_2 = read_data(
        result_dir,
        algorithm=algorithm_name_2,
    )

    if compare_all_combinations:
        # 批量对比 3*3*3 = 27 种情况
        compare_all_conditions(
            result_dir, algorithm_name_1, algorithm_name_2, max_rank_number=10
        )
    else:

        show_comparison_precision_recall(
            data_list_1, data_list_2, label_1=algorithm_name_1, label_2=algorithm_name_2
        )
        show_comparison_avg_metrics(
            data_list_1, data_list_2, label_1=algorithm_name_1, label_2=algorithm_name_2
        )
