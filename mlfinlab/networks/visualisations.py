"""
These methods allows the user to easily deploy graph visualisations given an input file dataframe.
"""

from mlfinlab.networks.dash_graph import DashGraph
from mlfinlab.networks.graph import MST
from mlfinlab.codependence import get_distance_matrix


def generate_mst_server(log_returns_df, mst_algorithm='kruskal', distance_matrix_type='angular', jupyter=False,
                        colours=None, sizes=None):
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
