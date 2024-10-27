#!/usr/bin/env python
# coding=utf-8
from common.util.data_reader.DatesetRead import DatasetReader


class AmericanCollegeFootball(DatasetReader):
    def __init__(self):
        # Raw Data Path
        super().__init__()
        self.data_path = r"./data/American-College-football/raw_data.txt"
        # Truth Table Path
        self.truthtable_path = r"./data/American-College-football/truth_table.txt"
        self.number_of_community = 12
        # name of dataset
        self.dataset_name = "AmericanCollegeFootball"

    def read_data(self):
        """
        Reads raw data representing edges between nodes.
        :return: list[[int, int]], Each element is [x, y], indicating an edge exists between nodes x and y.
        """
        content = []
        min_node = float('inf')  # Initialize the minimum node ID to a large value to find the smallest node ID

        with open(self.data_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Convert each line into a pair of integers [x, y]
                nodes = [int(num) for num in line.split()]
                content.append(nodes)

                # Update minimum node ID to ensure node indexing starts from 0
                min_node = min(min_node, *nodes)

        # If the minimum node ID is not 0, adjust all node IDs to start from 0
        if min_node != 0:
            content = [[node - min_node for node in pair] for pair in content]

        return content

    def read_truthtable(self):
        """
        Reads the truth table indicating community membership for each node.
        :return: list[[int, int]], Each element is [x, y], indicating node x belongs to community y.
        """
        content = []
        min_node = float('inf')  # Initialize the minimum node ID
        min_community = float('inf')  # Initialize the minimum community ID

        with open(self.truthtable_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Convert each line into a pair of integers [x, y]
                nodes = [int(num) for num in line.split()]
                content.append(nodes)

                # Update minimum node ID and community ID to start from 0
                min_node = min(min_node, nodes[0])
                min_community = min(min_community, nodes[1])

        # Adjust node and community IDs if they do not start from 0
        if min_node != 0 or min_community != 0:
            content = [[node - min_node, community - min_community] for node, community in content]

        return content

    def get_base_information(self):
        raw_data = self.read_data()
        truth_table = self.read_truthtable()

        # Display basic information about the dataset
        print("Number of edges:", len(raw_data))
        print("Number of nodes in truth table:", len(truth_table))
        print("Sample edges (raw_data):", raw_data[:5])
        print("Sample truth table entries:", truth_table[:5])
