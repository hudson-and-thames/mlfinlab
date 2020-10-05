.. _networks-minimum_spanning_tree:

.. note::
    This section includes an accompanying Jupyter Notebook Tutorial that is now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

.. note::

    Strategies were implemented with modifications from:

    1. `Huang, Feixue, Pengfei Gao, and Yu Wang. "Comparison of Prim and Kruskal on Shanghai and Shenzhen 300 Index hierarchical structure tree." 2009 International Conference on Web Information Systems and Mining. IEEE, 2009. <https://ieeexplore.ieee.org/abstract/document/5369459/>`_
    2. `Marti, Gautier, et al. "A review of two decades of correlations, hierarchies, networks and clustering in financial markets." (2017). <https://arxiv.org/abs/1703.00485>`_
    3. `Onnela, J-P., et al. "Dynamics of market correlations: Taxonomy and portfolio analysis." (2003) <https://www.researchgate.net/publication/8952753_Dynamics_of_Market_Correlations_Taxonomy_and_Portfolio_Analysis>`_

===========================
Minimum Spanning Tree (MST)
===========================

Network analysis can provide interesting insights into the dynamics of the market, and the continually changing behaviour.

A Minimum Spanning Trees (MST) is a useful method of analyzing complex networks, for aspects such as risk management, portfolio design, trading strategies.
For example Onnela et al. (2003) notices that the optimal Markowitz portfolio is found at the outskirts of the tree .
Analysing the Tree structure, as a representation of the market, can give us an idea about the stability and state of the market and predict how shocks will propagate through a network.

A Minimum Spanning Tree (MST) is a graph consisting of the fewest number of edges needed for all nodes to be connected by
some path - where the combination of edge weights sum to the smallest total possible.

The process of creating an MST is based on the Greedy algorithm, where the MST consists of `n` nodes and `n-1` edges.
The resulting MST ensures no loops, and that the combination of edges has the minimum weight.

Minimum Spanning Trees have the following properties (Marti et al. 2017):

 1. MST strongly shrinks during a stock crisis.
 2. The optimal Markowitz portfolio lies practically at all times on the outskirts of the tree.
 3. The normalized tree length and the investment diversification potential are very strongly correlated.

Further interesting features are noted:

 - Volatility shocks always start in the fringe and propagate inwards.
 - Emergence of an internal structure comprising multiple groups of strongly coupled components is a signature of market development.
 - The MST provides a taxonomy which is well compatible with the sector classification provided by an outside institution.

The two algorithms used to create the MST are Prim's and Kruskal's. In Prim's, the edges 'grow' from a single root node,
whereas Kruskal's considers every edge on its own. Prim's has time complexity :math:`O(n)` and Kruskal's has a time complexity
of :math:`O(n^2)` (Huang et al. 2009).

Based on experimental results by Huang et al. on 253 stocks from Shangai and Shenzhen 300,
Prim's used up 64,009 two dimension arrays, whereas Kruskal's used 31,878 one dimension arrays.

Due to the nature of the respective algorithms,
Prim's is recommended for sample sizes larger than 100, and Kruskal's for small sample sizes or when space complexity
is more important (Huang et al. 2009).

The ``MST`` class uses wrappers for `NetworkX library's
<https://networkx.github.io/>`_ functions for MST methods.

----

MST Algorithms
##############

Kruskal's Algorithm
*******************

The steps of Kruskal's algorithm:

1. Sort all the edges from smallest to largest.
2. Add the smallest edge to the final spanning tree. If adding the edge creates a cycle, move onto the next edge.
3. Keep adding all edges until all the nodes are a part of the tree.

.. figure:: images/data/KruskalDemo.gif
    :align: center

    Kruskal's algorithm `by Shiyu Ji - Own work, CC BY-SA 4.0, <https://commons.wikimedia.org/w/index.php?curid=54420893>`_.

Kruskal's algorithm is essentially the same as the Single Linkage Clustering method
which results in a hierarchical dendrogram. However, with the MST the aim is to display the links between elements
as opposed to clusters.

Kruskal's algorithm is commonly implemented with a disjoint set data structure:

.. code-block::

    Kruskals(G):

    # spanning tree to be returned A
    A = ∅

    For each vertex v ∈ G.V:
        MAKE-SET(v) # every vertex in its own set

    For each edge (u, v) ∈ G.E ordered by increasing order by weight(u, v):
        if FIND-SET(u) ≠ FIND-SET(v): # if u and v aren't in the same set
            A = A ∪ {(u, v)} # add the edge to A
            # merge the set containing u and the set containing v
            UNION(u, v)

    return A

More information on Kruskal's and Prim's algorthms can be found in the book "Introduction to algorithms" (2009) by Cormen, Leiserson, Rivest and Stein.

Prim's Algorithm
****************

The steps for Prim's algorithm are:

1. Choose and visit an arbitrary node.
2. Of all the edges from visited, pick the smallest and add the edge to the tree.
3. Keep adding edges, until all nodes are visited.

.. figure:: images/data/PrimAlgDemo.gif
    :align: center

    Prim's algorithm by `Shiyu Ji - Own work, CC BY-SA 4.0, <https://commons.wikimedia.org/w/index.php?curid=54420894>`_.

Prim's algorithm is recommended from a 100 vertices upwards for better time complexity (Huang et al 2009).
Compared to Kruskal's, Prim's does not calculate all the edges from shortest to largest, instead growing from a starting node, making it more
time-efficient for bigger data sets.

.. code-block::

    T = ∅; # T is the mst for final output

    U = { 1 }; # U is set of visited nodes

    while (U ≠ V)
        let (u, v) be the lowest cost edge such that u ∈ U and v ∈ V - U;
        T = T ∪ {(u, v)}
        U = U ∪ {v}

    return T

By default the `generate_mst_server` method uses Kruskal's algorithm. To use Prim's algorithm instead,
pass the parameter `mst_algorithm='prim'` into the `generate_mst_server` method. The documentation for
`generate_mst_server` can be found on the :ref:`Visualisations section <networks-visualisations>`.

--------------

MST with Custom Matrix
######################

To create a MST visualisation, it is best to use the :ref:`Visualisations methods <networks-visualisations>`.
The `generate_mst_server` takes in a dataframe of log returns and returns a server.
However, if you would like to input a custom matrix instead of a distance matrix, please read the information below.

To create an MST visualisation, you must input a matrix into the `MST` class.
The `DashGraph` class then takes in a `MST` as input and returns a server.

Implementation
**************

.. automodule:: mlfinlab.networks.mst

    .. autoclass:: MST
        :members:
        :inherited-members:

        .. automethod:: __init__

MST Algorithms
**************

When you initialise the MST object, the mst is generated and stored as attribute an `self.graph`.
Kruskal's algorithms is used as a default.

.. code-block::

    mst_graph = MST(custom_matrix, 'custom')

To use Prim's algorithm, pass the string 'prim'. Take a look below at the initialization method `__init__` to see how the algorithm is set.

.. code-block::

    mst_graph = MST(custom_matrix, 'custom', 'prim')

Example Code
************

.. code-block::

    import pandas as pd

    # Import MST class
    from mlfinlab.networks.mst import MST

    # Import Dash Graph class
    from mlfinlab.networks.dash_graph import DashGraph

    # Import file containing stock log returns
    log_return_dataframe = pd.read_csv('path_to_file', index_col=False)

    # Create your custom matrix
    correlation_matrix = log_return_dataframe.corr(method='pearson')

    # Create your custom matrix
    custom_matrix = mlfinlab.codependence.get_distance_matrix(correlation_matrix, distance_metric='angular')

    # Creates MST graph class objects from custom input matrix
    graph = MST(custom_matrix, 'custom')

    # Create and get the server
    dash_graph = DashGraph(graph)
    server = dash_graph.get_server()

    # Run server
    server.run_server()

Customizing the Graphs
**********************

To further customize the MST when it is displayed in the UI,
you can add colours and change the sizes to represent for example industry groups and market cap of stocks.
These features are optional.

Adding Colours to Nodes
***********************

The colours can be added by passing a dictionary of group name to list of node names corresponding to the nodes input.
You then pass the dictionary to the `set_node_groups` method.

.. code-block::

    # Optional - add industry groups for node colour
    industry = {"tech": ['NVDA', 'TWTR', 'MSFT'], "utilities": ['JNJ'], "automobiles": ['TSLA', 'GE']}
    graph.set_node_groups(industry)

Adding Sizes to Nodes
*********************

The sizes can be added in a similar manner, via a list of numbers which correspond to the node indexes.
The UI of the graph will then display the nodes indicating the different sizes.

.. code-block::

    # Optional - adding market cap for node size
    market_caps = [2000, 2500, 3000, 1000, 5000, 3500, 500, 1700]
    graph.set_node_size(market_caps)


----

Research Notebook
#################

.. note::
    This and other accompanying Jupyter Notebook Tutorials are now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

The following notebook provides a more detailed exploration of the MST creation.

* `MST visualisation`_

.. _`MST visualisation`: https://github.com/Hudson-and-Thames-Clients/research/tree/master/Networks/mst.ipynb
