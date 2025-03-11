import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 读取所有pkl文件
data_list = []
pkl_directory = (
    r"D:\code\FYP\CommunityDetection\algorithm\common\benchmark\generated_graphs"
)
pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith(".pkl")]

for file in pkl_files:
    file_path = os.path.join(pkl_directory, file)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        data_list.append(data)

# 初始化数据存储
num_nodes_list = []
num_edges_list = []
num_communities_list = []
average_degrees = []
graph_densities = []
graph_diameters = []

# 提取信息
for data in data_list:
    G = data["graph"]

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_communities = len(data["communities"])
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0  # 每个节点的平均度数
    density = nx.density(G)  # 计算图的密度

    # 计算图的直径（仅适用于连通图）
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        # diameter = 0
    else:
        diameter = None  # 记录 None，表示图不连通

    # 存储数据
    num_nodes_list.append(num_nodes)
    num_edges_list.append(num_edges)
    num_communities_list.append(num_communities)
    average_degrees.append(avg_degree)
    graph_densities.append(density)
    graph_diameters.append(diameter)

# 创建DataFrame
df = pd.DataFrame(
    {
        "Number of Nodes": num_nodes_list,
        "Number of Edges": num_edges_list,
        "Number of Communities": num_communities_list,
        "Average Degree": average_degrees,
        "Graph Density": graph_densities,
        "Graph Diameter": graph_diameters,
    }
)

# 绘制图表
plt.figure(figsize=(15, 10))

# 1. 节点数量分布
plt.subplot(3, 2, 1)
plt.hist(num_nodes_list, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Number of Nodes")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Nodes")

# 2. 边的数量分布
plt.subplot(3, 2, 2)
plt.hist(num_edges_list, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Number of Edges")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Edges")

# 3. 社区数量分布
plt.subplot(3, 2, 3)
plt.hist(num_communities_list, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Number of Communities")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Communities")

# 4. 平均度数分布
plt.subplot(3, 2, 4)
plt.hist(average_degrees, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Average Degree")
plt.ylabel("Frequency")
plt.title("Distribution of Average Degree")

# 5. 图的密度分布
plt.subplot(3, 2, 5)
plt.hist(graph_densities, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Graph Density")
plt.ylabel("Frequency")
plt.title("Distribution of Graph Density")

# 6. 图的直径分布（过滤 None 值）
valid_diameters = [d for d in graph_diameters if d is not None]
plt.subplot(3, 2, 6)
plt.hist(valid_diameters, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Graph Diameter")
plt.ylabel("Frequency")
plt.title("Distribution of Graph Diameter")

plt.tight_layout()
plt.show()
