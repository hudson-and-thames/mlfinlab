"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DualDashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash

class DualDashGraph:
    """
    The DualDashGraph class is the inerface for comparing and highlighting the difference between two graphs.
    Two Graph class objects should be supplied - such as MST and ALMST graphs.
    """

    def __init__(self, graph_one, graph_two, app_display='default'):
        """
        Initialises the dual graph interface and generates the interface layout.

        :param graph_one: (Graph) The first graph for the comparison interface.
        :param graph_two: (Graph) The second graph for the comparison interface.
        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.
        """

        pass

    @staticmethod
    def _select_other_graph_node(data, elements):
        """
        Callback function to select the other graph node when a graph node
        is selected by setting selected to True.

        :param data: (Dict) Dictionary of "tapped" or selected node.
        :param elements: (Dict) Dictionary of elements.
        :return: (Dict) Returns updates dictionary of elements.
        """

        pass

    def _generate_comparison_layout(self, graph_one, graph_two):
        """
        Returns and generates a dual comparison layout.

        :param graph_one: (Graph) The first graph object for the dual interface.
        :param graph_two: (Graph) Comparison graph object for the dual interface.
        :return: (html.Div) Returns a Div containing the interface.
        """

        pass

    @staticmethod
    def _get_default_stylesheet(weights):
        """
        Returns the default stylesheet for initialisation.

        :param weights: (List) A list of weights of the edges.
        :return: (List) A List of definitions used for Dash styling.
        """

        pass

    def _set_cyto_graph(self):
        """
        Updates and sets the two cytoscape graphs using the corresponding components.
        """

        pass

    def _update_elements_dual(self, graph, difference, graph_number):
        """
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param graph: (Graph) Graph object such as MST or ALMST.
        :param difference: (List) List of edges where the two graphs differ.
        :param graph_number: (Int) Graph number to update the correct graph.
        """

        pass

    def get_server(self):
        """
        Returns the comparison interface server

        :return: (Dash) Returns the Dash app object, which can be run using run_server.
            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.
        """

        pass
