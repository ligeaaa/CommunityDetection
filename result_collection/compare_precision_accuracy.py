import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re


# ------- 引入用户自定义的 read_data 函数 -------
def read_data(
    result_dir,
    algorithm=None,
    point_level=None,
    density_level=None,
    community_size_level=None,
):
    pkl_files = [f for f in os.listdir(result_dir) if f.endswith(".pkl")]
    filtered_files = []
    for file in pkl_files:
        match = re.match(
            r"([\w\-.]+)-(\d+)_point-(level\d), density-(level\d), community_size-(level\d)\.pkl",
            file,
        )
        if match:
            algo_name, point, point_lvl, density_lvl, community_size_lvl = (
                match.groups()
            )
            if (
                (algorithm and algorithm != algo_name)
                or (point_level and point_level != point_lvl)
                or (density_level and density_level != density_lvl)
                or (community_size_level and community_size_level != community_size_lvl)
            ):
                continue
            filtered_files.append(file)

    data_list = []
    for file in filtered_files:
        file_path = os.path.join(result_dir, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data["title"] = os.path.splitext(file)[0]
            data_list.append(data)
    return data_list


# ------- 评估指标计算逻辑 -------
def calculate_precision_accuracy(data_list):
    all_sorted_metrics = []
    max_community_count = 0
    for data in data_list:
        community_stats = data["metrics"]["acc_per_community"]["community_stats"]
        community_metrics = []
        for stats in community_stats.values():
            TP, FP, FN = stats["TP"], stats["FP"], stats["FN"]
            real_size = TP + FN
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / real_size if real_size > 0 else 0
            accuracy = (precision + recall) / 2
            community_metrics.append((real_size, precision, accuracy))
        community_metrics.sort(reverse=True, key=lambda x: x[0])
        max_community_count = max(max_community_count, len(community_metrics))
        all_sorted_metrics.append(community_metrics)
    return all_sorted_metrics, max_community_count


def compute_average_metrics(all_sorted_metrics, max_rank=10):
    aggregated_metrics = [[] for _ in range(max_rank)]
    rank_counts = [0] * max_rank
    for dataset_metrics in all_sorted_metrics:
        for rank, (_, precision, accuracy) in enumerate(dataset_metrics[:max_rank]):
            aggregated_metrics[rank].append((precision, accuracy))
            rank_counts[rank] += 1
    avg_results = []
    for rank, metrics in enumerate(aggregated_metrics):
        if metrics:
            avg_precision = np.mean([m[0] for m in metrics])
            avg_accuracy = np.mean([m[1] for m in metrics])
            avg_results.append(
                (rank + 1, avg_precision, avg_accuracy, rank_counts[rank])
            )
        else:
            avg_results.append((rank + 1, 0, 0, 0))
    return avg_results


def plot_community_bars(ax, x, counts):
    ax2 = ax.twinx()
    ax2.bar(x, counts, width=0.4, alpha=0.3, color="lightgray", label="Community Count")
    ax2.set_ylabel("Community Count", fontsize=11)
    ax2.set_ylim(0, max(counts) * 1.4)


def plot_metrics_with_bars(metrics_dict, metric_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "Leiden_Rarev0.03": ("-", "o"),
        "Leiden": ("--", "s"),
        "Louvain": (":", "x"),
    }
    for algo, results in metrics_dict.items():
        x = [r[0] for r in results]
        y = [r[1] if metric_name == "precision" else r[2] for r in results]
        # counts = [r[3] for r in results]
        linestyle, marker = styles.get(algo, ("-", "."))
        ax.plot(
            x,
            y,
            linestyle=linestyle,
            marker=marker,
            label=f"{algo.replace('Leiden_Rarev0.03', 'Leiden_Rare')} {metric_name.title()}",
        )
    x_sample = [r[0] for r in list(metrics_dict.values())[0]]
    bar_counts = [r[3] for r in list(metrics_dict.values())[0]]
    plot_community_bars(ax, x_sample, bar_counts)
    ax.set_xlabel("Community Rank")
    ax.set_ylabel(f"Average {metric_name.title()}")
    ax.set_title(f"Community Rank vs Average {metric_name.title()}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"average_{metric_name}.png")
    plt.show()


# ------- 主执行入口 -------
def main():
    result_dir = r"D:\code\FYP\CommunityDetection\result_collection\result\test"
    algorithms = ["Leiden_Rarev0.03", "Leiden", "Louvain"]
    metrics_dict = {}
    for algo in algorithms:
        data_list = read_data(result_dir, algorithm=algo)
        all_metrics, _ = calculate_precision_accuracy(data_list)
        avg_metrics = compute_average_metrics(all_metrics, max_rank=10)
        metrics_dict[algo] = avg_metrics
    plot_metrics_with_bars(metrics_dict, "precision")
    plot_metrics_with_bars(metrics_dict, "accuracy")


if __name__ == "__main__":
    main()
