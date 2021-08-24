"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import heapq
import itertools
from itertools import count
import warnings

import networkx as nx
from matplotlib import pyplot as plt

from mlfinlab.networks.graph import Graph


class PMFG(Graph):
    """
    PMFG class creates and stores the PMFG as an attribute.
    """

    def __init__(self, input_matrix, matrix_type):
        """
        PMFG class creates the Planar Maximally Filtered Graph and stores it as an attribute.

        :param input_matrix: (pd.Dataframe) Input distance matrix
        :param matrix_type: (str) Matrix type name (e.g. "distance").
        """

        pass

    def get_disparity_measure(self):
        """
        Getter method for the dictionary of disparity measure values of cliques.

        :return: (Dict) Returns a dictionary of clique to the disparity measure.
        """

        pass

    def _calculate_disparity(self):
        """
        Calculate disparity given in Tumminello M, Aste T, Di Matteo T, Mantegna RN.
        A tool for filtering information in complex systems.
        https://arxiv.org/pdf/cond-mat/0501335.pdf

        :return: (Dict) Returns a dictionary of clique to the disparity measure.
        """

        pass

    def _generate_cliques(self):
        """
        Generate cliques from all of the nodes in the PMFG.
        """

        pass

    def create_pmfg(self, input_matrix):
        """
        Creates the PMFG matrix from the input matrix of all edges.

        :param input_matrix: (pd.Dataframe) Input matrix with all edges
        :return: (nx.Graph) Output PMFG matrix
        """

        pass

    def get_mst_edges(self):
        """
        Returns the list of MST edges.

        :return: (list) Returns a list of tuples of edges.
        """

        pass

    def edge_in_mst(self, node1, node2):
        """
        Checks whether the edge from node1 to node2 is a part of the MST.

        :param node1: (str) Name of the first node in the edge.
        :param node2: (str) Name of the second node in the edge.
        :return: (bool) Returns true if the edge is in the MST. False otherwise.
        """

        pass

    def get_graph_plot(self):
        """
        Overrides parent get_graph_plot to plot it in a planar format.

        Returns the graph of the MST with labels.
        Assumes that the matrix contains stock names as headers.

        :return: (AxesSubplot) Axes with graph plot. Call plt.show() to display this graph.
        """

        pass
