.. _implementations-portfolio_optimisation:

========
Portfolio Optimisation
========

The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them. Next up, lets discuss about some of these implementations and the different parameters they require.

Hierarchical Risk Parity (HRP)
==============================

Hierarchical Risk Parity is a novel portfolio optimisation method developed by
Marcos Lopez de Prado. The working of the algorithm can be broken down into 3 steps:

1. Based on the expected returns of the assets, they are segregated into clusters via hierarchical
   tree clustering.
2. Based on these clusters, the covariance matrix of the returns is diagonalised in a quasi manner such that assets
   within the same cluster are regrouped together.
3. Finally, the weights are assigned to each cluster in a recursive manner. At each node, the weights are broken
   down into the sub-cluster until all the individual assets are assigned a unique weight.

Although, it is a simple algorithm, HRP has been found to be a very stable algorithm as compared to its older counterparts.
This is because, HRP does not involve taking inverse of the covariance matrix matrix which makes it robust to small changes
in the covariances of the asset returns. For a detailed explanation of how HRP works, we have written an excellent `blog post <https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/>`_ about it.

Implementation
~~~~~~~~~~~~~~

.. automodule:: mlfinlab.portfolio_optimization.hrp

    .. autoclass:: HierarchicalRiskParity
        :members:

        .. automethod:: __init__

The Critical Line Algorithm (CLA)
=================================

This is a robust alternative to the quadratic optimisation used to find mean-variance optimal portfolios. The major difference
between classic Mean-Variance and CLA is the type of optimisation problem solved. A typical mean-variance optimisation problem
looks something like this:

        :math:`\underset{w}{\text{minimise}} ~ \left\{w^T \Sigma w \right\}`

where, :math:`\sum_{i}w_{i} = 1` and :math:`0 <= w <= 1`. CLA also solves the same problem but with some added constraints - each weight of an asset in the portfolio can have different lower and upper bounds. The optimisation objective still remains the same but the second constraint changes to - :math:`l_{i} <= w_{i} <= u_{i}`. Each weight in the allocation has an upper and a lower bound, which increases the number of constraints to be solved.

The current CLA implementation in the package supports the following solution strings:

1. ``cla_turning_points`` : Calculates the set of CLA turning points. These are the original solution weights calculated the CLA algorithm.
2. ``max_sharpe`` : Calculates the weights relating to the maximum Sharpe Ratio portfolio.
3. ``min_volatility`` : Calculates the weights relating to Minimum Variance portfolio.
4. ``efficient_frontier`` : Calculates all weights in the efficient frontier(also includes the CLA turning points).

Implementation
~~~~~~~~~~~~~~

.. automodule:: mlfinlab.portfolio_optimization.cla

    .. autoclass:: CLA
        :members:

        .. automethod:: __init__

Mean-Variance Optimisation
==========================

This class contains some classic Mean-Variance optimisation techniques based on Harry Markowitz's methods. We use `cvxopt <https://cvxopt.org/>`_ as our quadratic optimiser instead of the more frequently used `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_. This was a design choice for two reasons: (a) the documentation of cvxopt is better than that of scipy and (b) cvxopt's code is much more readable and easier to understand.

Implementation
~~~~~~~~~~~~~~

.. automodule:: mlfinlab.portfolio_optimization.mean_variance

    .. autoclass:: MeanVarianceOptimisation
        :members:

        .. automethod:: __init__

Currently, the following solution strings are supported by MVO class:

1. ``inverse_variance`` : Calculates the weights according to simple inverse-variance allocation.
2. ``max_sharpe`` : Calculates the weights relating to the maximum Sharpe Ratio portfolio.
3. ``min_volatility`` : Calculates the weights relating to Minimum Variance portfolio.
4. ``efficient_risk`` : Calculates an efficient risk portfolio for a specified target return


Examples
=======

In this section, we provide some code snippets for new users to get started with the portfolio optimisation module.

Importing the Classes
~~~~~~~~~~~~~~

::

	from mlfinlab.portfolio_optimization.cla import CLA
	from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
	from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation
	import numpy as np
	import pandas as pd

Reading Data
~~~~~~~~~~~~~~

It is fairly straightforward to read the data using pandas and pass it to the public methods. Here, we read a csv file of historical stock prices.
::

	# Read in data
	stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date') # The date column may be named differently for your input.

.. note::

    We provide great flexibility to the users in terms of the input data - either they can pass raw historical stock prices as the parameter :py:mod:`asset_prices` in which case the expected returns and covariance matrix will be calculated using this data. Else, they can also pass pre-calculated :py:mod:`expected_returns` and :py:mod:`covariance_matrix`. For specific input types, please look at the doc-strings.


Allocating the Weights
~~~~~~~~~~~~~~

::

	# Compute HRP weights
	hrp = HierarchicalRiskParity()
	hrp.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, resample_by='B')
	hrp_weights = hrp.weights.sort_values(by=0, ascending=False, axis=1)

	# Compute IVP weights
	mvo = MeanVarianceOptimisation()
	mvo.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='inverse_variance', resample_by='B
	ivp_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1)

For HRP and IVP, you can access the computed weights as shown above. They are in the form of a dataframe and we can sort them in descending order of their weights.

::

    # Compute different solutions using CLA
    cla = CLA()

    # Turning Points
    cla.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices solution='cla_turning_points')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1) # Gives a dataframe with each row as a solution (turning_points)

    # Maximum Sharpe Solution
    cla.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='max_sharpe')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1) # Single set of weights for the max-sharpe portfolio
    max_sharpe_value = cla.max_sharpe # Accessing the max sharpe value

    # Minimum Variance Solution
    cla.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='min_volatility')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1) # Single set of weights for the min-variance portfolio
    min_variance_value = cla.min_var # Accessing the min-variance value

    # Efficient Frontier Solution
    cla.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='efficient_frontier')
    cla_weights = cla.weights
    means, sigma = cla.efficient_frontier_means, cla.efficient_frontier_sigma

Lets look at the MVO class and its different solutions,

::

    # Compute different mean-variance solutions using MVO
    mvo = MeanVarianceOptimisation()

    # Maximum Sharpe Solution
    mvo.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='max_sharpe')
    mvo_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1) # Single set of weights for the max-sharpe portfolio

    # Minimum Variance Solution
    mvo.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='min_volatility')
    mvo_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1) # Single set of weights for the min-variance portfolio

    # Efficient Risk Solution
    mvo.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, solution='efficient_risk', target_return=0.4)
    mvo_weights = mvo.weights

    # Portfolio Characteristics
    portfolio_return, sharpe_ratio, risk = mvo.portfolio_return, mvo.portfolio_sharpe_ratio, mvo.portfolio_risk


Plotting
~~~~~~~~~~~~~~

There are two plotting functions:

1. ``plot_clusters()`` : Plots the hierarchical clusters formed during the clustering step in HRP. This is visualised in the form of dendrograms - a very common way of visualising the hierarchical tree clusters.

::

    hrp = HierarchicalRiskParity()
    hrp.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, resample_by='B')
    hrp.plot_clusters(assets=stock_prices.columns)

.. image:: portfolio_optimisation_images/dendrogram.png

2. ``plot_efficient_frontier()`` : Plots the efficient frontier. The red dot corresponds to the Maximum Sharpe portfolio.

::

    mvo = MeanVarianceOptimisation()
    mvo.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices, resample_by='B')

    # Assuming there is a stock_returns dataframe
    mvo.plot_efficient_frontier(covariance=stock_returns.cov(),
                                expected_asset_returns=stock_returns.mean()*252,
                                num_assets=len(stock_returns.columns))

.. image:: portfolio_optimisation_images/efficient_frontier.png


Research Notebooks
==================

The following research notebooks can be used to better understand how the algorithms within this module can be used on real stock data.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Chapter16/Chapter16.ipynb









