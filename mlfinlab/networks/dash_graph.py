"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""

import json
import random

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from networkx import nx


class DashGraph:
    """
    This DashGraph class creates a server for Dash cytoscape visualisations.
    """

    def __init__(self, input_graph, app_display='default'):
        """
        Initialises the DashGraph object from the Graph class object.
        Dash creates a mini Flask server to visualise the graphs.

        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.
        :param input_graph: (Graph) Graph class from graph.py.
        """

        pass

    def _set_cyto_graph(self):
        """
        Sets the cytoscape graph elements.
        """

        pass

    def _get_node_group(self, node_name):
        """
        Returns the industry or sector name for a given node name.

        :param node_name: (str) Name of a given node in the graph.
        :return: (str) Name of industry that the node is in or "default" for nodes which haven't been assigned a group.
        """

        pass

    def _get_node_size(self, index):
        """
        Returns the node size for given node index if the node sizes have been set.

        :param index: (int) The index of the node.
        :return: (float) Returns size of node set, 0 if it has not been set.
        """

        pass

    def _update_elements(self, dps=4):
        """
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param dps: (int) Decimal places to round the edge values.
        """

        pass

    def _generate_layout(self):
        """
        Generates the layout for cytoscape.

        :return: (dbc.Container) Returns Dash Bootstrap Component Container containing the layout of UI.
        """

        pass

    def _assign_colours_to_groups(self, groups):
        """
        Assigns the colours to industry or sector groups by creating a dictionary of group name to colour.

        :param groups: (List) List of industry groups as strings.
        """

        pass

    def _style_colours(self):
        """
        Appends the colour styling to stylesheet for the different groups.
        """

        pass

    def _assign_sizes(self):
        """
        Assigns the node sizing by appending to the stylesheet.
        """

    pass

    def get_server(self):
        """
        Returns a small Flask server.

        :return: (Dash) Returns the Dash app object, which can be run using run_server.
            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.
        """

        pass

    @staticmethod
    def _update_cytoscape_layout(layout):
        """
        Callback function for updating the cytoscape layout.
        The useful layouts for MST have been included as options (cola, cose-bilkent, spread).

        :return: (Dict) Dictionary of the key 'name' to the desired layout (e.g. cola, spread).
        """

        pass

    def _update_stat_json(self, stat_name):
        """
        Callback function for updating the statistic shown.

        :param stat_name: (str) Name of the statistic to display (e.g. graph_summary).
        :return: (json) Json of the graph information depending on chosen statistic.
        """

        pass

    def get_graph_summary(self):
        """
        Returns the Graph Summary statistics.
        The following statistics are included - the number of nodes and edges, smallest and largest edge,
        average node connectivity, normalised tree length and the average shortest path.

        :return: (Dict) Dictionary of graph summary statistics.
        """

        pass

    def _round_decimals(self, dps):
        """
        Callback function for updating decimal places.
        Updates the elements to modify the rounding of edge values.

        :param dps: (int) Number of decimals places to round to.
        :return: (List) Returns the list of elements used to define graph.
        """

        pass

    def _get_default_stylesheet(self):
        """
        Returns the default stylesheet for initialisation.

        :return: (List) A List of definitions used for Dash styling.
        """

        pass

    def _get_toast(self):
        """
        Toast is the floating colour legend to display when industry groups have been added.
        This method returns the toast component with the styled colour legend.

        :return: (html.Div) Returns Div containing colour legend.
        """

        pass

    def _get_default_controls(self):
        """
        Returns the default controls for initialisation.

        :return: (dbc.Card) Dash Bootstrap Component Card which defines the side panel.
        """

        pass


class PMFGDash(DashGraph):
    """
    PMFGDash class, a child of DashGraph, is the Dash interface class to display the PMFG.
    """

    def __init__(self, input_graph, app_display='default'):
        """
        Initialise the PMFGDash class but override the layout options.
        """

        pass

    def _update_elements(self, dps=4):
        """
        Overrides the parent DashGraph class method _update_elements, to add styling for the MST edges.
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param dps: (int) Decimal places to round the edge values. By default, this will round to 4 d.p's.
        """

        pass

    def _get_default_stylesheet(self):
        """
        Gets the default stylesheet and adds the MST styling.

        :return: (List) Returns the stylesheet to be added to the graph.
        """

        pass
