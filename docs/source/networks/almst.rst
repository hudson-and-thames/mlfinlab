.. _networks-almst:

.. note::

   The following source elaborates extensively on the topic:

   `M. Tumminello, C. Coronnello, F. Lillo, S. Micciche, R. N. Mantegna, Spanning trees and bootstrap reliability estimation in correlation-based networks, International Journal of Bifurcation and Chaos 17 (2007) 2319–2329. <https://arxiv.org/pdf/physics/0605116.pdf>`_

=============================================
Average Linkage Minimum Spanning Tree (ALMST)
=============================================

Average Linkage Minimum Spanning Tree (ALMST) shows better results in recognising the economic sectors and sub-sectors than the  Minimum Spanning Tree (Tumminello et al. 2007).
The ALMST, as defined by `Tumminello et al. (2007), <https://arxiv.org/pdf/physics/0605116.pdf>`_ is a variation of the MST.

Just as the MST is associated with the Single Linkage Clustering Algorithm (SCLA), the ALMST is based on the UPGMA (unweighted pair group method with arithmetic mean) algorithm for hierarchal clustering.
This is also known as Average Linkage Cluster Algorithm (ALCA) method. The UPGMA method is frequently used in biology for phylogenetic tree as a diagrammatic representation of the evolutionary relatedness between organisms.
A step by step example of UPGMA or ALCA method for trees can be found `here. <https://en.wikipedia.org/wiki/UPGMA>`_

----

ALMST
#####

ALMST vs. MST
*************

Instead of choosing the next edge by the minimum edge (distance based MST), the next edge is chosen by the minimum *average* distance between two existing clusters.
Tumminello et al. (2007) give an example MST algorithm to show how the ALMST algorithm differs from the MST algorithm.

Where :math:`g` is the graph, :math:`S_{i}` is the set of vertices, :math:`n` is the number of elements, :math:`C` is the correlation matrix of elements :math:`ρ_{ij}`, connected component of a graph :math:`g` containing a given vertex :math:`i`.
The starting point of the procedure is an empty graph :math:`g` with :math:`n` vertices.

1. Set :math:`Q` as the matrix of elements :math:`q_{ij}` such that :math:`Q = C`, where :math:`C` is
   the estimated correlation matrix.
2. Select the maximum correlation :math:`q_{hk}` between elements belonging to different connected components :math:`S_{h}`
   and :math:`S_{k}` in :math:`g^{2}`.
3. Find elements :math:`u`, :math:`p` such that :math:`p_{up} = \max\{{\rho_{ij}}, \forall i \in S_{h}` and :math:`\forall j \in S_{k} \}`
4. Add to :math:`g` the link between elements :math:`u` and :math:`p` with weight :math:`\rho_{up}`. Once the link is added
   to :math:`g`, :math:`u` and :math:`p` will belong to the same connected component :math:`S = S_{h} \cup S_{k}`.
5. Redefine the matrix :math:`Q`:

.. math::

    \begin{cases}
    q_{ij} = q_{hk}, & \text{ if i} \in S_{h} \text{ and j} \in S_{k}

    q_{ij} = Max \{q_{pt}, p \in S \text{ and } t \in S_{j}, \text{ with } S_{j} \neq S\}, & \text{ if i} \in S \text{ and j } \in S_{j}

    q_{ij} = q_{ij}, & \text{ otherwise};
    \end{cases}

6. If :math:`g` is still a disconnected graph then go to step (2) else stop.

This is the case for a correlation matrix (taking the maximum correlation value for the MST edges).

However, for distance matrices, the MST algorithm orders the edges by the minimum distance, by replacing step (5) with:

.. math::

    \begin{cases}
    q_{ij} = q_{hk}, & \text{ if i} \in S_{h} \text{ and j} \in S_{k}

    q_{ij} = Min \{q_{pt}, p \in S \text{ and } t \in S_{j}, \text{ with } S_{j} \neq S\}, & \text{ if i} \in S \text{ and j } \in S_{j}

    q_{ij} = q_{ij}, & \text{ otherwise};
    \end{cases}

By replacing eq. in step (5) with

.. math::

    \begin{cases}
    q_{ij} = q_{hk}, & \text{ if i} \in S_{h} \text{ and j} \in S_{k}

    q_{ij} = Mean \{q_{pt}, p \in S \text{ and } t \in S_{j}, \text{ with } S_{j} \neq S\}, & \text{ if i} \in S \text{ and j } \in S_{j}

    q_{ij} = q_{ij}, & \text{ otherwise};
    \end{cases}

we obtain an algorithm performing the ALCA. We call the resulting tree :math:`g` an ALMST.

User Interface
**************

The ALMST can be generated in the same way as the MST, but by creating an `ALMST` class object instead.
Since `MST` and `ALMST` are both subclasses of `Graph`, `MST` and `ALMST` are the input to the initialisation method of class `Dash`.
However, the recommended way to create visualisations, is to use the methods from `visualisations` file, unless you would like to input a custom matrix.

Creating the `ALMST` visualisation is a similar process to creating the `MST`. You can replace:

.. code-block::

    mst = MST(input_matrix, 'input matrix type')

With:

.. code-block::

    almst = ALMST(input_matrix, 'input matrix type')

This creates the ALMST object containing `self.graph` as the graph of the ALMST. The `ALMST` object can then be inputted to the Dash interface.

Implementation
**************

.. automodule:: mlfinlab.networks.almst

    .. autoclass:: ALMST
        :members:
        :inherited-members:

        .. automethod:: __init__

ALMST Algorithms
****************

When you initialise the ALMST object, the ALMST is generated and stored as attribute an `self.graph`.
Kruskal's algorithms is used as a default.

.. code-block::

    almst_graph = ALMST(custom_matrix, 'custom')

To use Prim's algorithm, pass the string 'prim'.

.. code-block::

    almst_graph = ALMST(custom_matrix, 'custom', 'prim')

Example Code
************

.. code-block::

    import pandas as pd

    # Import ALMST class
    from mlfinlab.networks.almst import ALMST

    # Import Dash Graph class
    from mlfinlab.networks.dash_graph import DashGraph

    # Import file containing stock log returns
    log_return_dataframe = pd.read_csv('path_to_file', index_col=False)

    # Create your custom matrix
    correlation_matrix = log_return_dataframe.corr(method='pearson')

    # Create your custom matrix
    custom_matrix = mlfinlab.codependence.get_distance_matrix(correlation_matrix,
                                                              distance_metric='angular')

    # Creates MST graph class objects from custom input matrix
    graph = ALMST(custom_matrix, 'distance')

    # Create and get the server
    dash_graph = DashGraph(graph)
    server = dash_graph.get_server()

    # Run server
    server.run_server()

Customizing the Graphs
**********************

To further customize the ALMST when it is displayed in the Dash UI,
you can add colours and change the sizes to represent for example industry groups and market cap of stocks.

These features are optional and only work for the Dash interface (not the comparison interface).
As the comparison interface, highlights the central nodes (with degree greater than or equal to 5).

**Adding Colours to Nodes**

The colours can be added by passing a dictionary of group name to list of node names corresponding to the nodes input.
You then pass the dictionary to the `set_node_groups` method.

.. code-block::

    # Optional - add industry groups for node colour
    industry = {"tech": ['NVDA', 'TWTR', 'MSFT'], "utilities": ['JNJ'], "automobiles": ['TSLA', 'GE']}
    graph.set_node_groups(industry)

**Adding Sizes to Nodes**

The sizes can be added in a similar manner, via a list of numbers which correspond to the node indexes.
The UI of the graph will then display the nodes indicating the different sizes.

.. code-block::

    # Optional - adding market cap for node size
    market_caps = [2000, 2500, 3000, 1000, 5000, 3500, 500, 1700]
    graph.set_node_size(market_caps)

Comparison Interface
####################

.. figure::
    images/data/dualinterface.png

ALMST and MST can be compared easily using the `DualDashGraph` interface, where you can see the MST and ALMST side by side.
You can also create the dual interface using `generate_mst_almst_comparison` method in the :ref:`Visualisations <networks-visualisations>` section.
This is the recommended way as it reduces the number of steps needed to create the interface.

.. code-block::

    # Create MST graph
    mst = MST(input_matrix, 'input matrix type')

    # Create ALMST graph
    almst = ALMST(input_matrix, 'input matrix type')

    # Get DualDash server
    server = DualDashGraph(mst, almst)
    server.run_server()
