.. _portfolio_optimisation-hierarchical_risk_parity:

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

==============================
Hierarchical Risk Parity (HRP)
==============================

The Hierarchical Risk Parity algorithm is a novel portfolio optimisation method developed by Marcos Lopez de Prado. A quick
overview of the different steps in the algorithm:


    |h3| **1. Hierarchical Tree Clustering** |h3_|
    This step breaks down the assets in our portfolio into different hierarchical clusters using the famous Hierarchical Tree
    Clustering algorithm. The assets in the portfolio are segregated into clusters which mimic the real-life interactions between
    the assets in a portfolio - some stocks are related to each other more than others and hence can be grouped within the same
    cluster. At the end of the step, we are left with the follow tree structure (also called a dendrogram),

    .. image:: portfolio_optimisation_images/dendrogram.png

    |br|

    |h3| **2. Matrix Seriation** |h3_|
    Matrix seriation is a very old statistical technique which is used to rearrange the data to show the inherent clusters
    clearly. Using the order of hierarchical clusters from the previous step, we rearrange the rows and columns of the covariance
    matrix of stocks so that similar investments are placed together and dissimilar investments are placed far apart

    .. image:: portfolio_optimisation_images/seriation.png

    |br|

    |h3| **3. Recursive Bisection** |h3_|
    This is the final and the most important step of this algorithm where the actual weights are assigned to the assets in a
    top-down recursive manner. Based on the hierarchical tree dendrogram formed in the first step, the weights trickle down the
    tree and get assigned to the portfolio assets.


Although, it is a simple algorithm, it has been found to be very stable as compared to its older counterparts (the traditional
mean variance optimisation methods).

.. tip::
    * For a detailed explanation of how hierarchical risk parity works, we have written an excellent `blog post <https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/>`_ about it.


Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.hrp

    .. autoclass:: HierarchicalRiskParity
        :members:

        .. automethod:: __init__

.. tip::
    |h4| Using Custom Distance Matrix |h4_|
    The hierarchical clustering step in the algorithm uses a distance matrix to calculate the clusters and form the hierarchical
    tree. By default, we use the distance matrix mentioned in the original paper,

    .. math::

      D(i, j) = \sqrt{\frac{1}{2} * (1 - \rho(i, j))}

    However, users can specify their own custom matrix to be used instead of the default one by passing an :math:`NxN` symmetric
    pandas dataframe or a numpy matrix using the :py:mod:`distance_matrix` parameter.

    |h4| Constructing a Long/Short Portfolio |h4_|

    |h4| Different Linkage Methods |h4_|
    HRP, by default, uses the single-linkage clustering algorithm. (See the tip under the HCAA algorithm for more details.)



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

    # Building a dollar neutral Long/Short portfolio by shorting the first 4 stocks and being long the others
    hrp = HierarchicalRiskParity()
    side_weights = pd.Series([1]*stock_prices.shape[1], index=self.data.columns)
    side_weights.loc[stock_prices.columns[:4]] = -1
    hrp.allocate(asset_prices=self.data, asset_names=self.data.columns, side_weights=side_weights)
    hrp.allocate(asset_prices=stock_prices, side_weights=side_weights, resample_by='B')
    hrp_weights = hrp.weights.sort_values(by=0, ascending=False, axis=1)


.. note::

    We provide great flexibility to the users in terms of the input data - either they can pass raw historical stock prices
    as the parameter :py:mod:`asset_prices` in which case the expected returns and covariance matrix will be calculated
    using this data. Else, they can also pass pre-calculated :py:mod:`expected_returns` and :py:mod:`covariance_matrix`.
    Ultimately, they can pass their own :py:mod:`distance_matrix` but the :py:mod:`covariance_matrix` is still requested
    for computing the clustered variances. The :py:mod:`linkage` method for the clustering part is also a parameter of the
    algorithm, default being single linkage.


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

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Machine%20Learning%20Asset%20Allocation/Chapter16.ipynb
