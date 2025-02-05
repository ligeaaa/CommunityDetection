#!/usr/bin/env python
# coding=utf-8
from collections import Counter, defaultdict

from algorithm.common.util.data_reader.DatesetRead import DatasetReader


class AmazonDataset(DatasetReader):
    """
    - Only the top 5,000 communities in this dataset were selected.
    - Since the original data contains overlapping communities,
    each node belonging to multiple communities can be assigned
    to the community with the largest membership.
    - For some nodes without a clear community affiliation, we chose to remove them.
    """

    def __init__(self):
        # Initialize the superclass
        super().__init__()
        # Paths to the data files
        self.data_path = r"./data/amazon/com-amazon.ungraph.txt"
        self.truthtable_path = r"./data/amazon/com-amazon.top5000.cmty.txt"
        # Number of communities
        self.number_of_community = 1399
        self.dataset_name = "AmazonDataset"

        # Preprocess data to filter and map nodes with community affiliation
        self.raw_data_with_community, self.node_mapping = self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocesses data to:
        1. Collect nodes with community affiliations from the community file.
        2. Map these nodes to a consecutive integer sequence starting from 0.
        3. Filter edges in the graph file to only include nodes with community affiliations.

        :return: A tuple (filtered_edges, node_mapping)
        """
        # Step 1: Read community file to get nodes with community affiliations
        node_with_community = set()
        with open(self.truthtable_path, "r", encoding="utf-8") as file:
            for line in file:
                community = [int(num) for num in line.split()]
                node_with_community.update(community)  # Add nodes to the set

        # Step 2: Create a mapping from original node IDs to consecutive IDs starting from 0
        node_mapping = {
            node: idx for idx, node in enumerate(sorted(node_with_community))
        }

        # Step 3: Filter edges in the graph file
        filtered_edges = []
        with open(self.data_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("#"):
                    continue

                # Convert each line's content to integers
                nodes = [int(num) for num in line.split()]

                # Only include edges where both nodes have community affiliation
                if nodes[0] in node_with_community and nodes[1] in node_with_community:
                    # Map nodes to consecutive IDs and store the edge
                    filtered_edges.append(
                        [node_mapping[nodes[0]], node_mapping[nodes[1]]]
                    )

        return filtered_edges, node_mapping

    def read_data(self):
        """
        Returns preprocessed and filtered graph data.
        :return: list of [a, b], where an undirected edge exists between product a and product b.
        """
        return self.raw_data_with_community

    def read_truthtable(self):
        """
        Reads the community data file and assigns each node to the largest community it belongs to.
        Ensures both node and community IDs are consecutive, starting from 0.
        :return: list of [int, int], where each element is [x, y] indicating node x belongs to community y.
        """
        node_to_communities = defaultdict(list)
        community_list = []  # Collect all communities as lists of nodes

        # Step 1: Parse the community file and assign nodes to communities
        with open(self.truthtable_path, "r", encoding="utf-8") as file:
            for line in file:
                community = [int(num) for num in line.split()]
                community_list.append(community)  # Add community to list

                # Add each node to its respective community list
                for node in community:
                    node_to_communities[node].append(
                        len(community_list) - 1
                    )  # Use index as community ID

        # Step 2: Assign each node to the largest community it belongs to using the precomputed node mapping
        node_community_pairs = []
        for node, community_ids in node_to_communities.items():
            most_common_community = Counter(community_ids).most_common(1)[0][0]
            if (
                node in self.node_mapping
            ):  # Only include nodes with community affiliation
                mapped_node = self.node_mapping[node]
                node_community_pairs.append((mapped_node, most_common_community))

        # Step 3: Map community IDs to a consecutive sequence starting from 0
        unique_communities = sorted(
            {community for _, community in node_community_pairs}
        )
        community_mapping = {
            old_id: new_id for new_id, old_id in enumerate(unique_communities)
        }

        # Step 4: Apply the community mapping to node-community pairs
        truth_table = [
            [node, community_mapping[community]]
            for node, community in node_community_pairs
        ]

        return truth_table

    def get_base_information(self):
        """
        Print basic information about the dataset.
        """
        graph_data = self.read_data()
        community_data = self.read_truthtable()
        num_nodes = len({node for edge in graph_data for node in edge})
        num_edges = len(graph_data)
        num_communities = len(set([community for _, community in community_data]))

        print(f"Dataset Name: {self.dataset_name}")
        print(f"Number of Nodes: {num_nodes}")
        print(f"Number of Edges: {num_edges}")
        print(f"Number of Communities: {num_communities}")


# Example usage:
# dataset = AmazonDataset()
# graph_data = dataset.read_data()
# community_data = dataset.read_truthtable()
# dataset.get_base_information()
