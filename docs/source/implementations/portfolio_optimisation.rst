.. _implementations-portfolio_optimisation:

========
Portfolio Optimisation
========

The portfolio optimisation module contains some classic algorithms that are used for asset allocation and optimising strategies. We will discuss these algorithms in detail below.

Hierarchical Risk Parity (HRP)
==============================

Hierarchical Risk Parity is a novel portfolio optimisation method developed by
Marcos Lopez de Prado. The working of the algorithm can be broken down into 3 steps:

1. Based on the expected returns of the assets, they are segregated into clusters via hierarchical
   tree clustering.
2. Based on these clusters, the covariance matrix of the returns is diagonalised in a quasi manner such that assets
   within the same cluster are regrouped together.
3. Finally, using an iterative approach, weights are assigned to each cluster recursively. At each node, the weight breaks
   down into the sub-cluster until all the individual assets are assigned a unique weight.

Although, it is a simple algorithm, HRP has been found to be a very stable algorithm as compared to its older counterparts.
This is because, HRP does not involve taking inverse of the covariance matrix matrix which makes it robust to small changes
in the covariances of the asset returns.

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

The current CLA implementation in the package supports the following solutions:

1. CLA Turning Points
2. Maximum Sharpe Portfolio
3. Minimum Variance Portfolio
4. Efficient Frontier Solution

Implementation
~~~~~~~~~~~~~~

.. automodule:: mlfinlab.portfolio_optimization.cla

    .. autoclass:: CLA
        :members:

        .. automethod:: __init__

Mean-Variance Optimisation
==========================

This class contains the classic Mean-Variance optimisation techniques which use quadratic optimisation to get solutions to the portfolio allocation problem. Currently, it only supports the basic inverse-variance allocation strategy (IVP) but we aim to add more functions to tackle different optimisation objectives like maximum sharpe, minimum volatility, targeted-risk return maximisation and much more.

Implementation
~~~~~~~~~~~~~~

.. automodule:: mlfinlab.portfolio_optimization.mean_variance

    .. autoclass:: MeanVarianceOptimisation
        :members:

        .. automethod:: __init__

Examples
=======

Lets see how to import and use the 3 portfolio optimisation classes

::

	from mlfinlab.portfolio_optimization.cla import CLA
	from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
	from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation
	import numpy as np
	import pandas as pd

::

	# Read in data
	stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date') # The date column may be named differently for your input.

One important thing to remember here - all the 3 classes require that the stock prices dataframe be indexed by date because internally this will be used to calculate the expected returns.

::

	# Compute HRP weights
	hrp = HierarchicalRiskParity()
	hrp.allocate(asset_prices=stock_prices, resample_by='B')
	hrp_weights = hrp.weights.sort_values(by=0, ascending=False, axis=1)

	# Compute IVP weights
	mvo = MeanVarianceOptimisation()
	mvo.allocate(asset_prices=stock_prices, solution='inverse_variance', resample_by='B
	ivp_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1)

For HRP and IVP, you can access the computed weights as shown above. They are in the form of a dataframe and we can sort them in descending order of their weights.

::

    # Compute different solutions using CLA
    cla = CLA()

    # Turning Points
    cla.allocate(asset_prices=stock_prices, resample_by='W', solution='cla_turning_points')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1) # Gives a dataframe with each row as a solution (turning_points)

    # Maximum Sharpe Solution
    cla.allocate(asset_prices=stock_prices, resample_by='W', solution='max_sharpe')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1) # Single set of weights for the max-sharpe portfolio
    max_sharpe_value = cla.max_sharpe # Accessing the max sharpe value

    # Minimum Variance Solution
    cla.allocate(asset_prices=stock_prices, resample_by='W', solution='min_volatility')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1) # Single set of weights for the min-variance portfolio
    min_variance_value = cla.min_var # Accessing the min-variance value

    # Efficient Frontier Solution
    cla.allocate(asset_prices=stock_prices, resample_by='W', solution='efficient_frontier')
    cla_weights = cla.weights
    means, sigma = cla.efficient_frontier_means, cla.efficient_frontier_sigma

Research Notebooks
==================

The following research notebooks can be used to better understand how the algorithms within this module can be used on real stock data.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Chapter16/Chapter16.ipynb









