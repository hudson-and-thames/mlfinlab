.. _portfolio_optimisation-critical_line_algorithm:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.


=================================
The Critical Line Algorithm (CLA)
=================================

This is a robust alternative to the quadratic optimisation used to find mean-variance optimal portfolios. The major difference
between classic Mean-Variance and CLA is the type of optimisation problem solved. A typical mean-variance optimisation problem
looks something like this:

.. math::

      \underset{w}{\text{minimise}} ~ \left\{w^T \Sigma w \right\}

Where, :math:`\sum_{i}w_{i} = 1` and :math:`0 <= w <= 1`. CLA also solves the same problem but with some added constraints - each weight
of an asset in the portfolio can have different lower and upper bounds. The optimisation objective still remains the same but the second
constraint changes to - :math:`l_{i} <= w_{i} <= u_{i}`. Each weight in the allocation has an upper and a lower bound, which increases
the number of constraints to be solved.

The current CLA implementation in the package supports the following solution strings:

1. ``cla_turning_points`` : Calculates the set of CLA turning points. These are the original solution weights calculated the CLA algorithm.
2. ``max_sharpe`` : Calculates the weights relating to the maximum Sharpe Ratio portfolio.
3. ``min_volatility`` : Calculates the weights relating to Minimum Variance portfolio.
4. ``efficient_frontier`` : Calculates all weights in the efficient frontier(also includes the CLA turning points).


Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.cla

    .. autoclass:: CLA
        :members:

        .. automethod:: __init__


Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.cla import CLA

    # Read in data
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute different solutions using CLA
    cla = CLA()

    # Turning : each row as a solution (turning_points)
    cla.allocate(asset_prices=stock_prices solution='cla_turning_points')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1)

    # Maximum Sharpe Solution
    cla.allocate(asset_prices=stock_prices, solution='max_sharpe')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1)
    max_sharpe_value = cla.max_sharpe # Accessing the max sharpe value

    # Minimum Variance Solution
    cla.allocate(asset_prices=stock_prices, solution='min_volatility')
    cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1)
    min_variance_value = cla.min_var # Accessing the min-variance value

    # Efficient Frontier Solution
    cla.allocate(asset_prices=stock_prices, solution='efficient_frontier')
    cla_weights = cla.weights
    means, sigma = cla.efficient_frontier_means, cla.efficient_frontier_sigma

.. note::

    We provide great flexibility to the users in terms of the input data - either they can pass raw historical stock prices
    as the parameter :py:mod:`asset_prices` in which case the expected returns and covariance matrix will be calculated
    using this data. Else, they can also pass pre-calculated :py:mod:`expected_returns` and :py:mod:`covariance_matrix`.


Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the algorithm as outlined at the back of Ch16 in
Advances in Financial Machine Learning.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Chapter16/Chapter16.ipynb
