"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""

import heapq
import itertools
from itertools import count

import networkx as nx
import numpy as np
import pandas as pd
from mlfinlab.networks.graph import Graph


class ALMST(Graph):
    """
    ALMST is a subclass of Graph which creates a ALMST Graph object.
    The ALMST class converts a distance matrix input into a ALMST matrix. This is then used to create a nx.Graph object.
    """

    def __init__(self, matrix, matrix_type, mst_algorithm='kruskal'):
        """
        Initialises the ALMST and sets the self.graph attribute as the ALMST graph.

        :param matrix: (pd.Dataframe) Input matrices such as a distance or correlation matrix.
        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").
        :param mst_algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim'.
            By default, MST algorithm uses Kruskal's.
        """

        pass

    @staticmethod
    def create_almst_kruskals(matrix):
        """
        This method converts the input matrix into a ALMST matrix.

        ! Currently only works with distance input matrix

        :param matrix: (pd.Dataframe) Input matrix.
        :return: (pd.Dataframe) ALMST matrix with all other edges as 0 values.
        """

        pass

    @staticmethod
    def _generate_ordered_heap(matrix, clusters):
        """
        Given the matrix of edges, and the list of clusters, generate a heap ordered by the average distance between the clusters.

        :param matrix: (pd.Dataframe) Input matrix of the distance matrix.
        :param clusters: (List) A list of clusters, where each list contains a list of nodes within the cluster.
        :return: (Heap) Returns a heap ordered by the average distance between the clusters.
        """

        pass

    @staticmethod
    def _calculate_average_distance(matrix, clusters, c_x, c_y):
        """
        Given two clusters, calculates the average distance between the two.

        :param matrix: (pd.Dataframe) Input matrix with all edges.
        :param clusters: (List) List of clusters.
        :param c_x: (int) Cluster x, where x is the index of the cluster.
        :param c_y: (int) Cluster y, where y is the index of the cluster.
        """

        pass

    @staticmethod
    def _get_min_edge(node, cluster, matrix):
        """
        Returns the minimum edge tuple given a node and a cluster.

        :param node: (str) String of the node name.
        :param cluster: (list) List of node names.
        :param matrix: (pd.DataFrame) A matrix of all edges.
        :return: (tuple) A tuple of average distance from node to the cluster, and the minimum edge nodes, i and j.
        """

        pass

    @staticmethod
    def _get_min_edge_clusters(cluster_one, cluster_two, matrix):
        """
        Returns a tuple of the minimum edge and the average length for two clusters.

        :param cluster_one: (list) List of node names.
        :param cluster_two: (list) List of node names.
        :param matrix: (pd.DataFrame) A matrix of all edges.
        :return: (tuple) A tuple of average distance between the clusters, and the minimum edge nodes, i and j.
        """

        pass

    @staticmethod
    def create_almst(matrix):
        """
        Creates and returns a ALMST given an input matrix using Prim's algorithm.

        :param matrix: (pd.Dataframe) Input distance matrix of all edges.
        :return: (pd.Dataframe) Returns the ALMST in matrix format.
        """

        pass

    @staticmethod
    def _add_next_edge(visited, children, matrix, almst_matrix):
        """
        Adds the next edge with the minimum average distance.

        :param visited: (Set) A set of visited nodes.
        :param children: (Set) A set of children or frontier nodes, to be visited.
        :param matrix: (pd.Dataframe) Input distance matrix of all edges.
        :param almst_matrix: (pd.Dataframe) The ALMST matrix.

        :return: (Tuple) Returns the sets visited and children, and the matrix almst_matrix.
        """

        pass
