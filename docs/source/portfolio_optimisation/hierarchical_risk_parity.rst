.. _portfolio_optimisation-hierarchical_risk_parity:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==============================
Hierarchical Risk Parity (HRP)
==============================

Hierarchical Risk Parity is a novel portfolio optimisation method developed by Marcos Lopez de Prado. The working of the
algorithm can be broken down into 3 steps:

1. Based on the expected returns of the assets, they are segregated into clusters via hierarchical tree clustering.
2. Based on these clusters, the covariance matrix of the returns is diagonalised in a quasi manner such that assets
   within the same cluster are regrouped together.
3. Finally, the weights are assigned to each cluster in a recursive manner. At each node, the weights are broken down
   into the sub-cluster until all the individual assets are assigned a unique weight.

Although, it is a simple algorithm, HRP has been found to be very stable as compared to its older counterparts.
This is because, HRP does not involve taking inverse of the covariance matrix matrix which makes it robust to small changes
in the covariances of the asset returns.

.. tip::
    * For a detailed explanation of how HRP works, we have written an excellent `blog post <https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/>`_ about it.
    * HRP uses the single-linkage clustering algorithm. (See the tip under the HCAA algorithm for more details.)


Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.hrp

    .. autoclass:: HierarchicalRiskParity
        :members:

        .. automethod:: __init__


Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity

    # Read in data
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute HRP weights
    hrp = HierarchicalRiskParity()
    hrp.allocate(asset_prices=stock_prices, resample_by='B')
    hrp_weights = hrp.weights.sort_values(by=0, ascending=False, axis=1)


.. note::

    We provide great flexibility to the users in terms of the input data - either they can pass raw historical stock prices
    as the parameter :py:mod:`asset_prices` in which case the expected returns and covariance matrix will be calculated
    using this data. Else, they can also pass pre-calculated :py:mod:`expected_returns` and :py:mod:`covariance_matrix`.


Plotting
########

``plot_clusters()`` : Plots the hierarchical clusters formed during the clustering step in HRP. This is visualised in the form of dendrograms - a very common way of visualising the hierarchical tree clusters.

.. code-block::

    # Instantiate HRP Class
    hrp = HierarchicalRiskParity()
    hrp.allocate(asset_prices=stock_prices, resample_by='B')

    # Plot Dendrogram
    hrp.plot_clusters(assets=stock_prices.columns)

.. image:: portfolio_optimisation_images/dendrogram.png


Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the algorithm as outlined at the back of Ch16 in
Advances in Financial Machine Learning.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Chapter16/Chapter16.ipynb
