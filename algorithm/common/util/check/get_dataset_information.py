#!/usr/bin/env python
# coding=utf-8
from collections import defaultdict

from loguru import logger

# 配置 Loguru 控制台输出，添加颜色格式
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def validate_input(raw_data, truth_table):
    """
    Validates the format of raw_data and truth_table inputs, ensuring proper community and node numbering.
    """
    logger.info("Starting input validation...")

    errors_found = False  # 标记是否有错误，用于记录每条错误

    # Validate format of raw_data
    if not isinstance(raw_data, list) or not all(
        isinstance(edge, list)
        and len(edge) == 2
        and all(isinstance(node, int) for node in edge)
        for edge in raw_data
    ):
        logger.error("Invalid raw_data format: Expected list of [int, int] pairs.")
        errors_found = True

    # Validate format of truth_table
    if not isinstance(truth_table, list) or not all(
        isinstance(entry, list)
        and len(entry) == 2
        and all(isinstance(value, int) for value in entry)
        for entry in truth_table
    ):
        logger.error("Invalid truth_table format: Expected list of [int, int] pairs.")
        errors_found = True

    # Check community ID continuity
    community_ids = sorted({community for _, community in truth_table})
    if community_ids == list(range(len(community_ids))):
        logger.info("Community IDs are consecutive from 0.")
    else:
        logger.error(
            f"Community IDs are not consecutive from 0. Found communities: {community_ids}"
        )
        errors_found = True

    # Check node ID continuity
    node_ids_in_edges = sorted({node for edge in raw_data for node in edge})
    if node_ids_in_edges == list(range(node_ids_in_edges[-1] + 1)):
        logger.info("Node IDs are consecutive from 0.")
    else:
        logger.error(
            f"Node IDs are not consecutive from 0. Found nodes in edges: {node_ids_in_edges}"
        )
        errors_found = True

    # Check if all nodes are assigned to a community
    node_ids_in_truth_table = {node for node, _ in truth_table}
    unassigned_nodes = set(node_ids_in_edges) - node_ids_in_truth_table
    if unassigned_nodes:
        logger.warning(
            f"{len(unassigned_nodes)} nodes are not assigned to any community: {sorted(unassigned_nodes)}"
        )
    else:
        logger.info(
            "All nodes in raw_data are assigned to at least one community in truth_table."
        )

    # Check for nodes in truth_table not present in raw_data
    extra_nodes = node_ids_in_truth_table - set(node_ids_in_edges)
    if extra_nodes:
        logger.warning(
            f"Nodes in truth_table but not in raw_data: {sorted(extra_nodes)}"
        )
    else:
        logger.info("All nodes in truth_table are present in raw_data.")

    if not errors_found:
        logger.info("Input validation completed successfully.")
    return not errors_found  # 返回False如果发现错误，否则True


def is_overlapping_community(truth_table):
    """
    Determines whether the given community structure in `truth_table` represents overlapping communities.
    """
    node_to_communities = defaultdict(set)
    for node, community in truth_table:
        node_to_communities[node].add(community)

    for communities in node_to_communities.values():
        if len(communities) > 1:
            return True
    return False


def display_community_overlap_info(truth_table):
    """
    Determines and displays whether the community structure in `truth_table` represents overlapping communities.
    """
    is_overlapping = is_overlapping_community(truth_table)
    if is_overlapping:
        logger.info("The community structure is overlapping.")

        # Count nodes based on community membership
        node_to_communities = defaultdict(set)
        for node, community in truth_table:
            node_to_communities[node].add(community)

        community_counts = defaultdict(int)
        for communities in node_to_communities.values():
            count = len(communities)
            community_counts[count] += 1

        logger.info("Detailed community membership breakdown:")
        for membership_count, count in sorted(community_counts.items()):
            logger.info(
                f"Nodes belonging to {membership_count} community/communities: {count}"
            )
    else:
        logger.info(
            "The community structure is non-overlapping, with each node belonging to only one community."
        )


def get_dataset_info(raw_data, truth_table):
    """
    Calculate the number of nodes, edges, and communities from raw data and truth table.
    """
    nodes_in_edges = set()
    for edge in raw_data:
        nodes_in_edges.update(edge)
    num_nodes = len(nodes_in_edges)
    num_edges = len(raw_data)
    communities = {community for _, community in truth_table}
    num_communities = len(communities)

    logger.info(
        f"Dataset info - Nodes: {num_nodes}, Edges: {num_edges}, Communities: {num_communities}"
    )
    return {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Number of Communities": num_communities,
    }


def check_multiple_edges(raw_data):
    """
    Checks if there are multiple edges between any two nodes in `raw_data`.
    Outputs the nodes and the number of duplicate edges, if any.
    """
    edge_count = defaultdict(int)
    for edge in raw_data:
        node_pair = tuple(sorted(edge))
        edge_count[node_pair] += 1

    multiple_edges = {pair: count for pair, count in edge_count.items() if count > 1}

    if multiple_edges:
        logger.warning("Multiple edges detected between nodes:")
        for nodes, count in multiple_edges.items():
            logger.warning(f"Nodes {nodes[0]} and {nodes[1]} have {count} edges.")
    else:
        logger.info("No multiple edges found between any nodes.")


def analyze_dataset(raw_data, truth_table):
    """
    Combines and outputs information about the dataset's structure and community overlap,
    and checks for any multiple edges between nodes.
    """
    # Validate inputs
    if validate_input(raw_data, truth_table):
        logger.info("Inputs are valid.")

    # Display basic dataset info
    dataset_info = get_dataset_info(raw_data, truth_table)
    for key, value in dataset_info.items():
        logger.info(f"{key}: {value}")

    # Display community overlap information
    display_community_overlap_info(truth_table)

    # Check for multiple edges
    check_multiple_edges(raw_data)


if __name__ == "__main__":
    # Sample data
    raw_data = [
        [0, 1],
        [1, 2],
        [1, 2],  # Duplicate edge: repeated edge between node 1 and node 2
        [
            2,
            1,
        ],  # Duplicate edge (reversed): repeated edge between node 1 and node 2 in reverse order
        [3, 4],
        [3, 4, 1],  # Format error: expected [int, int] but contains three nodes
        [1, 6],
        [3, 4],  # Duplicate edge: repeated edge between node 3 and node 4
    ]

    truth_table = [
        [0, 1],
        [1, 1],
        [2, 0],
        [3, 0],
        [4, 2],
        [
            5,
            1,
        ],  # Node in truth_table but missing in raw_data: node 5 has no associated edges in raw_data
    ]

    # Call the function to analyze the dataset
    analyze_dataset(raw_data, truth_table)
