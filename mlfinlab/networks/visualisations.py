"""
These methods allows the user to easily deploy graph visualisations given an input file dataframe.
"""

import warnings
import networkx as nx

from mlfinlab.networks.dash_graph import DashGraph, PMFGDash
from mlfinlab.networks.dual_dash_graph import DualDashGraph
from mlfinlab.networks.mst import MST
from mlfinlab.networks.almst import ALMST
from mlfinlab.networks.pmfg import PMFG
from mlfinlab.codependence import get_distance_matrix


def generate_mst_server(log_returns_df, mst_algorithm='kruskal', distance_matrix_type='angular',
                        jupyter=False, colours=None, sizes=None):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param mst_algorithm: (str) A valid MST type such as 'kruskal', 'prim', or 'boruvka'.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
        corresponding to the node indexes inputted in the initial dataframe.
    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
        in the initial dataframe.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """

    pass


def create_input_matrix(log_returns_df, distance_matrix_type):
    """
    This method returns the distance matrix ready to be inputted into the Graph class.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :return: (pd.Dataframe) A dataframe of a distance matrix.
    """

    pass


def generate_almst_server(log_returns_df, distance_matrix_type='angular',
                          jupyter=False, colours=None, sizes=None):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
        corresponding to the node indexes inputted in the initial dataframe.
    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
        in the initial dataframe.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """

    pass


def generate_mst_almst_comparison(log_returns_df, distance_matrix_type='angular', jupyter=False):
    """
    This method returns a Dash server ready to be run.

    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
        with stock names as columns.
    :param distance_matrix_type: (str) A valid sub type of a distance matrix,
        namely 'angular', 'abs_angular', 'squared_angular'.
    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
    :return: (Dash) Returns the Dash app object, which can be run using run_server.
        Returns a Jupyter Dash object if the parameter jupyter is set to True.
    """

    pass


def generate_pmfg_server(log_returns_df, input_type='distance',
                         jupyter=False, colours=None, sizes=None):
    """
      This method returns a PMFGDash server ready to be run.

      :param log_returns_df: (pd.Dataframe) An input dataframe of log returns
          with stock names as columns.
      :param input_type: (str) A valid input type correlation or distance. Inputting correlation will add the edges
          by largest to smallest, instead of smallest to largest.
      :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.
      :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes
          corresponding to the node indexes inputted in the initial dataframe.
      :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted
          in the initial dataframe.
      :return: (Dash) Returns the Dash app object, which can be run using run_server.
          Returns a Jupyter Dash object if the parameter jupyter is set to True.
      """

    pass


def generate_central_peripheral_ranking(nx_graph):
    """
    Given a NetworkX graph, this method generates and returns a ranking of centrality.
    The input should be a distance based PMFG.

    The ranking combines multiple centrality measures to calculate an overall ranking of how central or peripheral the
    nodes are.
    The smaller the ranking, the more peripheral the node is. The larger the ranking, the more central the node is.

    The factors contributing to the ranking include Degree, Eccentricity, Closeness Centrality, Second Order Centrality,
    Eigen Vector Centrality and Betweenness Centrality. The formula for these measures can be found on the NetworkX
    documentation (https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html)

    :param nx_graph: (nx.Graph) NetworkX graph object. You can call get_graph() on the MST, ALMST and PMFG to retrieve
        the nx.Graph.
    :return: (List) Returns a list of tuples of ranking value to node.
    """

    pass