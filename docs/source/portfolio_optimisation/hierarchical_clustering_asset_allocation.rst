.. _portfolio_optimisation-hierarchical_clustering_asset_allocation:

.. |br| raw:: html

    <br>

.. |h3| raw:: html

    <h3>

.. |h3_| raw:: html

    </h3>

.. |h4| raw:: html

    <h4>

.. |h4_| raw:: html

    </h4>

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

===============================================
Hierarchical Clustering Asset Allocation (HCAA)
===============================================

The Hierarchical Risk Parity algorithm focuses on allocation of risk using a hierarchical clustering approach and using the
variance of the clusters to allocate weights. While variance is a very simple and popular representation of risk used in the
investing world, it is not the optimal one and can underestimate the true risk of a portfolio which is why there are many other
important risk metrics used by investment managers that can correctly reflect the true risk of a portfolio/asset. With respect to
this, the original HRP algorithm can be tweaked to allocate its weights based on different risk representations of the clusters
and generate better weights. This class implements an improved hierarchical clustering algorithm which gives the option of using
the following metrics:

1. ``minimum_variance`` : Variance of the clusters is used as a risk metric.
2. ``minimum_standard_deviation`` : Standard deviation of the clusters is used as a risk metric.
3. ``sharpe_ratio`` : Sharpe ratio of the clusters is used as a risk metric.
4. ``equal_weighting`` : All clusters are weighed equally in terms of risk.
5. ``expected_shortfall`` : Expected shortfall (CVaR) of the clusters is used as a risk metric.
6. ``conditional_drawdown_at_risk`` : Conditional drawdown at risk (CDaR) of the clusters is used as a risk metric.

.. tip::
    |h4| Underlying Literature |h4_|
    This implementation is based on the following two papers written by Thomas Raffinot.
        * **Hierarchical Clustering based Asset Allocation:** `Raffinot, Thomas, The Hierarchical Equal Risk Contribution Portfolio (August 23, 2018) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540>`_
        * **Hierarchical Equal Risk Contribution:**

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.hcaa

    .. autoclass:: HierarchicalClusteringAssetAllocation
        :members:

        .. automethod:: __init__

.. note::
    |h4| Using Custom Input |h4_|
    We provide great flexibility to the users in terms of the input data - they can either pass their own pre-calculated input
    matrices/dataframes or leave it to us to calculate them. A quick reference on common input parameters which you will encounter
    throughout the portfolio optimisation module:
        * :py:mod:`asset_prices`: Dataframe/matrix of historical raw asset prices **indexed by date**.
        * :py:mod:`asset_returns`: Dataframe/matrix of historical asset returns. This will be a :math:`TxN` matrix where :math:`T` is the time-series and :math:`N` refers to the number of assets in the portfolio.
        * :py:mod:`expected_asset_returns`: List of expected returns per asset i.e. the mean of historical asset returns. This refers to the parameter :math:`\mu` used in portfolio optimisation literature. For a portfolio of 5 assets, ``expected_asset_returns = [0.45, 0.56, 0.89, 1.34, 2.4]``.
        * :py:mod:`covariance_matrix`: The covariance matrix of asset returns.


.. tip::
    **What are the differences between the 3 Linkage Algorithms?**

    The following is taken directly from and we highly recommend you read:

    `Papenbrock, J., 2011. Asset Clusters and Asset Networks in Financial Risk Management and Portfolio Optimization (Doctoral
    dissertation, Karlsruher Institut für Technologie (KIT)). <https://d-nb.info/1108447864/34>`_

    **1. Single-Linkage**

    The idea behind single-linkage is to form groups of elements, which have the smallest distance to each other (nearest
    neighbouring clustering). This oftentimes leads to large groups/chaining.

    The single-link algorithm oftentimes forms clusters that are chained together and leaves large clusters. It can probably
    be best understood as a way to give a "more robust" estimation of the distance matrix and furthermore preserves the original
    structure as much as possible. Elements departing early from the tree can be interpreted as "different" from the overall dataset.
    In terms of application, the single-link clustering algorithm is very useful to gain insights in the correlation structure
    between assets and separates assets that were very different from the rest. If this separation is preferred and high weights
    should be put on "outliers" the single link certainly is a good choice.

    **2. Complete-Linkage**

    The complete-linkage algorithm tries to avoid those large groups by considering the largest distances between elements.
    It is thus called the farthest neighbour clustering.

    The complete-link algorithm has a different idea: elements should be grouped together in a way that they are not too
    different from each other when merged in a cluster. It thus has a much stronger definition of "similar pair of clusters".
    The complete-link algorithm therefore seems suitable for investors interested in grouping stocks that are similar in one cluster.

    **3. Average-Linkage**

    The average-linkage algorithm is a compromise between the single-linkage and complete-linkage algorithm.
