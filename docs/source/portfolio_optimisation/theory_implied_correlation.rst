.. _portfolio_optimisation-theory_implied_correlation:


================================
Theory-Implied Correlation (TIC)
================================
This TIC class includes algorithm to calculate the Theory-Implied Correlation and a method to calculate the correlation matrix distance proposed by Herdin and Bonek. This distance may be used to measure to which extent the TIC matrix has blended theory-implied views (tree structure of the elements) with empirical evidence (correlation matrix).

About the Algorithm
###################

The TIC algorithm is aiming to estimate a forward-looking correlation matrix based on economic theory. The method is using a theoretical classification of assets (hierarchical structure) and fits the empirical correlation matrix to the theoretical structure.

According to Lopez de Prado, author of the algorithm: "A problem of empirical correlation matrices is that they are purely observation driven, and do not impose a structural view of the investment universe, supported by economic theory."

Using the TIC approach allows to include forward-looking views to the world, instead of only backward-looking views in empirical correlations matrix.

The economic theory in the algorithm is represented in terms of a tree graph. The tree can include any number of levels needed, each branch should have one or more leaves, some branches may include more levels than other.

An example of how the theoretical structure can be used is the MSCI's Global Industry Classification Standard (GICS) for investments. Using this structure, each stock can be classified using four levels of depth.

To use a tree as the input to the algorithm, it should have the bottom-up order of the columns, where the leftmost column is corresponding to terminal leaves and the rifhtmost colums corresponding to the tree's root. An  example of a tree graph according to the GICS:

+---------------+----------------+-----------+-----------------+---------+
| Ticker        | Sub-Industry   | Industry  | Industry Group  | Sector  |
+===============+================+===========+=================+=========+
| A UN Equity   | 35203010       | 352030    | 3520            | 35      |
+---------------+----------------+-----------+-----------------+---------+
| AAL UW Equity | 20302010       | 203020    | 2030            | 20      |
+---------------+----------------+-----------+-----------------+---------+
| AAP UN Equity | 25504050       | 255040    | 2550            | 25      |
+---------------+----------------+-----------+-----------------+---------+
| AAPL UW Equity| 45202030       | 452020    | 4520            | 45      |
+---------------+----------------+-----------+-----------------+---------+

The empirical correlation matrix used in the TIC algorithm is estimated on historical observations. It should be symmetric and have a main diagonal of 1s. But it doesn't need to be invertible, positive-definite or non-singular.

Steps of the Algorithm
######################

The TIC algorithm consists of three steps:

1. On the first step, the theoretical tree graph structure of the assets is fit on the evidence presented by the empirical correlation matrix. The result of the first step is a binary tree (dendrogram) that sequentially clusters two elements together, while measuring how closely together the two elements are, until all elements are subsumed within the same cluster.

2. On the second step, a correlation matrix is derived from the linkage object. Each cluster in the global linkage object is decomposed to two elements, which can be either atoms or other clusters. Then the off -diagonal correlation between two elements are calculated based on the distances between them.

3. On the third step, the correlation matrix is de-noised. This is done by fitting the Marcenko-Pastur distribution to the eigenvalues of the matrix, calculating the maximum theoretical eigenvalue as a threshold and eliminating the eigenvalues higher than a set threshold. This algorithm is implemented in the RiskEstimators class.

.. tip::

    This algorithm is described in more detail in the work **Estimation of Theory-Implied Correlation Matrices** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/abstract_id=3484152>`_.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.tic

    .. autoclass:: TIC
        :members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.tic import TIC

    # Reading data
    tree_classification = pd.read_csv('TREE_FILE_PATH')
    stock_prices = pd.read_csv('DATA_FILE_PATH', parse_dates=True, index_col='Date')

    # Calculating the empirical correlation matrix
    corr_matrix = stock_prices.corr()

    # Calculating the relation of sample length T to the number of variables N
    # It's used for de-noising the TIC matrix
    tn_relation = stock_prices.shape[0] / stock_prices.shape[1]

    # The class that contains the TIC algorithm
    tic = TIC()

    # Calculating the Theory-Implied Correlation matrix
    tic_matrix = tic.tic_correlation(tree_classification, corr_matrix, tn_relation, kde_bwidth=0.01)

    # Calculating the distance between the empirical and the theory-implied correlation matrices
    matrix_distance = tic.corr_dist(corr_matrix, tic_matrix)

Research Notebooks
==================

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Theory-Implied Correlation Notebook`_

.. _Theory-Implied Correlation Notebook: https://github.com/hudson-and-thames/research/blob/master/TIC/TIC.ipynb

