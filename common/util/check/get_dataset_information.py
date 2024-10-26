#!/usr/bin/env python
# coding=utf-8

def is_overlapping_community(truth_table):
    """
    Determines whether the given community structure in `truth_table` represents overlapping communities.
    """
    node_to_communities = {}
    for node, community in truth_table:
        if node not in node_to_communities:
            node_to_communities[node] = set()
        node_to_communities[node].add(community)

    for communities in node_to_communities.values():
        if len(communities) > 1:
            return True
    return False


def display_community_overlap_info(truth_table):
    """
    Determines and displays whether the community structure in `truth_table` represents overlapping communities.
    If overlapping, provides detailed info about the number of nodes belonging to each count of communities.
    """
    is_overlapping = is_overlapping_community(truth_table)

    if is_overlapping:
        print("The community structure is overlapping.")
        node_to_communities = {}
        for node, community in truth_table:
            if node not in node_to_communities:
                node_to_communities[node] = set()
            node_to_communities[node].add(community)

        community_counts = {}
        for communities in node_to_communities.values():
            count = len(communities)
            community_counts[count] = community_counts.get(count, 0) + 1

        print("Detailed community membership breakdown:")
        for membership_count in sorted(community_counts):
            print(f"Nodes belonging to {membership_count} community/communities: {community_counts[membership_count]}")
    else:
        print("The community structure is non-overlapping, with each node belonging to only one community.")


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

    return {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Number of Communities': num_communities
    }


def check_multiple_edges(raw_data):
    """
    Checks if there are multiple edges between any two nodes in `raw_data`.
    Outputs the nodes and the number of duplicate edges, if any.

    :param raw_data: list of [int, int], where each [x, y] represents an edge between nodes x and y.
    """
    edge_count = {}
    for edge in raw_data:
        # Ensure the node pair is sorted to handle undirected edges
        node_pair = tuple(sorted(edge))
        edge_count[node_pair] = edge_count.get(node_pair, 0) + 1

    multiple_edges = {pair: count for pair, count in edge_count.items() if count > 1}

    if multiple_edges:
        print("Multiple edges detected between nodes:")
        for nodes, count in multiple_edges.items():
            print(f"Nodes {nodes[0]} and {nodes[1]} have {count} edges.")
    else:
        print("No multiple edges found between any nodes.")


def analyze_dataset(raw_data, truth_table):
    """
    Combines and outputs information about the dataset's structure and community overlap,
    and checks for any multiple edges between nodes.
    """
    # Display basic dataset info
    dataset_info = get_dataset_info(raw_data, truth_table)
    print("Dataset Information:")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    # Display community overlap information
    display_community_overlap_info(truth_table)

    # Check for multiple edges
    check_multiple_edges(raw_data)


if __name__ == '__main__':
    # Sample data
    raw_data = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 2],  # Duplicate edge for testing
        [2, 1],  # Duplicate edge in reverse for testing
    ]
    truth_table = [
        [0, 1],
        [1, 1],
        [2, 0],
        [3, 0],
        [4, 2]
    ]

    # Call the combined function to analyze dataset
    analyze_dataset(raw_data, truth_table)
