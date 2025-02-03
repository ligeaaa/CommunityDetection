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


# def get_pos()

def draw_communities(G, pos, communities=None, max_nodes=200, draw_networkx_labels=True):
    total_nodes = G.number_of_nodes()

    # 如果节点数超过 max_nodes，则随机选择一部分节点
    if total_nodes > max_nodes:
        nodes = set()
        # 如果有 communities 则优先保证每个社区有一定数量的节点
        if communities:
            for community in communities:
                sample_size = min(len(community), max_nodes // len(communities))  # 确保每个社区至少有一些节点
                nodes.update(random.sample(community, sample_size))

        # 补充随机节点直到达到 max_nodes
        remaining_nodes = max_nodes - len(nodes)
        if remaining_nodes > 0:
            remaining_random_nodes = random.sample(list(set(G.nodes()) - nodes), remaining_nodes)
            nodes.update(remaining_random_nodes)

        # 生成子图
        subgraph = G.subgraph(nodes)
    else:
        # 如果节点数不超过 max_nodes，则直接使用全部节点
        subgraph = G

    if communities:
        # 筛选出子图中的社区节点
        subgraph_communities = [[node for node in community if node in subgraph] for community in communities]
        # 使用 itertools 循环分配颜色
        colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])

        # 绘制不同社区的节点
        for community, color in zip(subgraph_communities, colors):
            nx.draw_networkx_nodes(subgraph, pos, nodelist=community, node_color=color, node_size=300, alpha=0.7)
    else:
        # 如果没有社区划分，则绘制原始图
        nx.draw_networkx_nodes(subgraph, pos, node_size=300, alpha=0.7, node_color='lightblue')

    # 绘制所有边
    nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), alpha=0.2)

    # 绘制节点标签
    if draw_networkx_labels == True:
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color='black')

    # 显示图
    plt.show()
