#!/usr/bin/env python
# coding=utf-8
from algorithm.common.util.data_reader.DatesetRead import DatasetReader


class PolbooksDataset(DatasetReader):
    def __init__(self):
        # Paths for raw data and truth table
        super().__init__()
        self.data_path = r"./data/political_books/raw_data.txt"
        self.truthtable_path = r"./data/political_books/truthtable.txt"

        # Define the number of communities
        self.number_of_community = 3  # three communities: neutral, conservative, and liberal
        # name of dataset
        self.dataset_name = "PolbooksDataset"

    def read_data(self):
        """
        Read raw edge data from file
        :return: list of [int, int], each entry [a, b] indicates an edge between nodes a and b
        """
        content = []
        min_node = float('inf')  # initialize minimum node value to infinity

        with open(self.data_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip header row if necessary
                if line.startswith("Source"):
                    continue

                # Split line content and extract nodes
                numbers = line.split()[:2]  # Only take the first two columns (Source, Target)
                numbers = [int(num) for num in numbers]
                content.append(numbers)

                # Track the minimum node number
                min_node = min(min_node, *numbers)

        # Normalize node numbers to start from 0
        if min_node != 0:
            content = [[num - min_node for num in pair] for pair in content]

        return content

    def read_truthtable(self):
        """
        Read truth table data
        :return: list of [int, int], where each element [node_id, community_id] indicates node's community membership
        """
        # Mapping for community labels to integers
        community_map = {"neutral": 0, "conservative": 1, "liberal": 2}
        content = []
        min_node = float('inf')  # Initialize min node to handle normalization

        with open(self.truthtable_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip header row
                if line.startswith("Id"):
                    continue

                # Split line content, map community to integer, and collect the data
                parts = line.split()
                node_id = int(parts[0])  # Node ID
                community_label = parts[-1].strip()  # Community label
                community_id = community_map.get(community_label, -1)  # Map label to community ID

                if community_id != -1:
                    content.append([node_id, community_id])

                # Track minimum node ID for normalization
                min_node = min(min_node, node_id)

        # Normalize node IDs if they do not start from 0
        if min_node != 0:
            content = [[node_id - min_node, community_id] for node_id, community_id in content]

        return content

    def get_base_information(self):
        """
        Print basic information for this dataset, including the number of nodes, edges, and communities.
        """
        print(f"Dataset Path: {self.data_path}")
        print(f"Truth Table Path: {self.truthtable_path}")
        print(f"Number of Communities: {self.number_of_community}")