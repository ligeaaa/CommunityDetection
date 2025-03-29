import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from algorithm.common.util.save_pkl import read_pkl_from_temp

# 结果文件目录
pkl_directory = (
    r"D:\code\FYP\CommunityDetection\algorithm\common\benchmark\generated_graphs"
)

# 存储所有社区的 size 和 diameter
community_sizes = []
community_diameters = []

if __name__ == "__main__":
    # 获取目录下所有的 .pkl 文件
    # pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]
    # size_to_diameters = defaultdict(list)
    # for file in pkl_files:
    #     print(file)
    #     file_path = os.path.join(pkl_directory, file)
    #     with open(file_path, "rb") as f:
    #         data = pickle.load(f)
    #
    #     G = data["graph"]
    #     communities = data["communities"]  # dict: node -> community_id
    #
    #     for community in communities:
    #         try:
    #             diam = nx.diameter(G.subgraph(community))
    #         except Exception:
    #             continue
    #
    #         size_to_diameters[len(community)].append(diam)
    #
    # save_pkl_to_temp(size_to_diameters, "truthtable_diameter")
    size_to_diameters = read_pkl_from_temp("truthtable_diameter")
    # 计算平均直径
    sizes = sorted(size_to_diameters.keys())
    avg_diameters = [np.mean(size_to_diameters[size]) for size in sizes]

    # 拟合函数：对数增长模型
    def log_func(x, a, b):
        return a * np.log(x) + b

    # 执行拟合
    popt, pcov = curve_fit(log_func, sizes, avg_diameters)

    # 生成拟合曲线
    x_fit = np.linspace(min(sizes), max(sizes), 500)
    y_fit = log_func(x_fit, *popt)
    # 可视化原始点 + 拟合曲线
    plt.figure(figsize=(8, 5))
    plt.scatter(sizes, avg_diameters, alpha=0.6, label="Average Diameter")
    plt.plot(
        x_fit, y_fit, color="red", label=f"Fitted: {popt[0]:.2f}·log(n) + {popt[1]:.2f}"
    )
    plt.title("Community Size vs Average Diameter with Logarithmic Fit")
    plt.xlabel("Community Size")
    plt.ylabel("Average Diameter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
