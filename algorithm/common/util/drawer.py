#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: drawer.py
@date: 2024/10/4 22:59
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
import itertools
import random
import networkx as nx
from matplotlib import pyplot as plt


def draw_communities(
    G,
    pos,
    communities=None,
    max_nodes=500,
    draw_networkx_labels=True,
    title=None,
    metrics=None,
):
    # 按照社区大小从大到小排序
    communities = sorted(communities, key=len, reverse=True)
    total_nodes = G.number_of_nodes()

    # 如果节点数超过 max_nodes，则随机选择一部分节点
    if total_nodes > max_nodes:
        nodes = set()
        if communities:
            for community in communities:
                sample_size = min(len(community), max_nodes // len(communities))
                nodes.update(random.sample(community, sample_size))

        remaining_nodes = max_nodes - len(nodes)
        if remaining_nodes > 0:
            remaining_random_nodes = random.sample(
                list(set(G.nodes()) - nodes), remaining_nodes
            )
            nodes.update(remaining_random_nodes)

        subgraph = G.subgraph(nodes)
    else:
        subgraph = G

    fig, ax = plt.subplots()

    legend_handles = []  # **存储图例**

    if communities:
        subgraph_communities = [
            [node for node in community if node in subgraph]
            for community in communities
        ]

        # **使用更多颜色，增强区分度**
        color_cycle = itertools.cycle(
            [
                "r",
                "g",
                "b",
                "c",
                "m",
                "y",
                "k",
                "orange",
                "purple",
                "brown",
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
        )

        # **绘制不同社区的节点**
        for i, (community, color) in enumerate(zip(subgraph_communities, color_cycle)):
            scatter = nx.draw_networkx_nodes(
                subgraph,
                pos,
                nodelist=community,
                node_color=color,
                node_size=300,
                alpha=0.7,
                ax=ax,
            )
            legend_handles.append((scatter, f"{i + 1}"))  # **存储 (句柄, 标签)**

    else:
        # **如果没有社区划分，则绘制原始图**
        scatter = nx.draw_networkx_nodes(
            subgraph, pos, node_size=300, alpha=0.7, node_color="lightblue", ax=ax
        )
        legend_handles.append((scatter, "未划分社区"))

    # **绘制所有边**
    nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), alpha=0.2, ax=ax)

    # **绘制节点标签**
    if draw_networkx_labels:
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color="black", ax=ax)

    # **设置标题**
    if title:
        ax.set_title(f"{title}")

    # **添加图例**
    handles, labels = zip(*legend_handles)  # **解压 legend 数据**
    ax.legend(
        handles,
        labels,
        loc="best",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.7,
    )

    # **在图的右侧显示 metrics 信息**
    if metrics:
        metrics_text = "\n".join(
            [
                (
                    f"{key}: {value:.4f}"
                    if not isinstance(value, dict)
                    else f"{key}: (dict skipped)"
                )
                for key, value in metrics.items()
            ]
        )

        plt.gcf().text(
            0.6,
            0.25,
            metrics_text,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6),
        )

    plt.show()

    return fig  # **返回 Matplotlib Figure 对象**


def draw_shortest_paths(G, pos):
    """计算并绘制不同长度的最短路径"""
    # 计算所有点对最短路径长度
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # 统计路径长度
    edge_color_map = {}  # 存储边及其颜色
    length_categories = {}  # 存储不同长度的边集合

    # 遍历所有点对，收集边及其对应的路径长度
    for node1 in shortest_paths:
        for node2, length in shortest_paths[node1].items():
            if node1 != node2 and (node2, node1) not in edge_color_map:
                if length not in length_categories:
                    length_categories[length] = []
                length_categories[length].append((node1, node2))

    # 分配颜色
    colors = itertools.cycle(
        ["r", "g", "b", "c", "m", "y", "k", "orange", "purple", "brown"]
    )
    length_color_map = {
        length: next(colors) for length in sorted(length_categories.keys())
    }

    # 记录每条边的颜色
    for length, edges in length_categories.items():
        for edge in edges:
            edge_color_map[edge] = length_color_map[length]

    # 绘制图
    plt.figure(figsize=(8, 8))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=300, alpha=0.7)

    # 绘制边（按最短路径长度分颜色）
    for length, edges in length_categories.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=length_color_map[length],
            alpha=0.6,
            width=2,
            label=f"Length {length}",
        )

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # 图例
    plt.legend()
    plt.title("Shortest Paths with Different Lengths")
    plt.show()
