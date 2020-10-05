"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

from abc import ABC

import networkx as nx
from matplotlib import pyplot as plt


class Graph(ABC):
    """
    This Graph class is a parent class for different types of graphs such as a MST.
    """

    def __init__(self, matrix_type):
        """
        Initializes the Graph object and the Graph class attributes.
        This includes the specific graph such as a MST stored as an attribute.

        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").
        """

        pass

    def get_matrix_type(self):
        """
        Returns the matrix type set at initialisation.

        :return: (str) String of matrix type (eg. "correlation" or "distance").
        """

        pass

    def get_graph(self):
        """
        Returns the Graph stored as an attribute.

        :return: (nx.Graph) Returns a NetworkX graph object.
        """

        pass

    def get_difference(self, input_graph_two):
        """
        Given two Graph with the same nodes, return a set of differences in edge connections.

        :param input_graph_two: (Graph) A graph to compare self.graph against.
        :return: (List) A list of unique tuples showing different edge connections.
        """

        pass

    def get_pos(self):
        """
        Returns the dictionary of the nodes coordinates.

        :return: (Dict) Dictionary of node coordinates.
        """

        pass

    def get_graph_plot(self):
        """
        Returns the graph of the MST with labels.
        Assumes that the matrix contains stock names as headers.

        :return: (AxesSubplot) Axes with graph plot. Call plt.show() to display this graph.
        """

        pass

    def set_node_groups(self, industry_groups):
        """
        Sets the node industry group, by taking in a dictionary of industry group to a list of node indexes.

        :param industry_groups: (Dict) Dictionary of the industry name to a list of node indexes.
        """

        pass

    def set_node_size(self, market_caps):
        """
        Sets the node sizes, given a list of market cap values corresponding to node indexes.

        :param market_caps: (List) List of numbers corresponding to node indexes.
        """

        pass

    def get_node_sizes(self):
        """
        Returns the node sizes as a list.

        :return: (List) List of numbers representing node sizes.
        """

        pass

    def get_node_colours(self):
        """
        Returns a map of industry group matched with list of nodes.

        :return: (Dict) Dictionary of industry name to list of node indexes.
        """

        pass
