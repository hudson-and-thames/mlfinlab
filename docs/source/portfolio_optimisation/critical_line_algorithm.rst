.. _portfolio_optimisation-critical_line_algorithm:

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

.. |h5| raw:: html

    <h5>

.. |h5_| raw:: html

    </h5>

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies.
    Each algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight
    allocations on the specific user data. This way, each implementation can be called in the same way and this makes it simple
    for users to use them.

=================================
The Critical Line Algorithm (CLA)
=================================

This is a robust alternative to the quadratic optimisation used to find mean-variance optimal portfolios. The major difference
between classic mean-variance optimisation and the critical line algorithm (CLA) are the optimisation constraints involved. A
typical mean-variance optimisation problem looks something like this:

.. math::
        \begin{align*}
            & \underset{\mathbf{w}}{\text{min}} & & w^T\sum w \\
            & \text{s.t.} & & \sum_{i=1}^{n}w_{i} = 1 \\
        \end{align*}

where :math:`w` refers to the set of weights for the portfolio assets, :math:`\sum` is the covariance matrix of the assets
and :math:`\mu` is the expected asset returns. CLA also solves the same problem but with some added constraints - each weight
of an asset in the portfolio can have different lower and upper bounds. The optimisation objective still remains the same but with
the new constraints on the weights, the problem looks like this:

.. math::
        \begin{align*}
            & \underset{\mathbf{w}}{\text{min}} & & w^T\sum w \\
            & \text{s.t.} & & \sum_{i=1}^{n}w_{i} = 1 \\
            &&& w_{i} <= u_{i} \\
            &&& w_{i} >= l_{i} \\
        \end{align*}

Each weight in the allocation has an upper and a lower bound, which increases the number of constraints to be solved.

.. tip::
   |h4| Underlying Literature |h4_|

   The following sources elaborate extensively on the topic:

   - **An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization** *by* David H. Bailey *and* Marcos Lopez de Prado `available here <https://papers.ssrn.com/sol3/abstract_id=2197616>`_.

.. note::
    |h4| Important Points about CLA |h4_|

    * It is the only algorithm specifically designed for inequality-constrained portfolio optimization problems, which guarantees that the exact solution is found after a given number of iterations
    * It does not only compute a single portfolio, but also derives the entire efficient frontier solution.

Supported Solutions
###################

MlFinLab's :py:mod:`CriticalLineAlgorithm` class provides the following solutions to be used out-of-the-box.

CLA Turning Points
******************

As described above, in a CLA problem there are some weights which are bounded by upper and lower bounds. These set of weights are
called *bounded* weights and the other weights not constrained by any bounds are called *free* weights.

A solution vector :math:`w^{*}` is a *turning point* its vicinity there is another solution vector with different free weights. This is important because in those regions of the solution space away from turning points the inequality constraints are effectively irrelevant with respect to the free assets.

This solution finds all the sets of weights which satisfy the definition of a *turning point*. This means that there can be multiple set of optimal weights (or turning points) for a problem.

**Solution String:** ``cla_turning_points``

Maximum Sharpe Ratio
********************

The optimal weights will be chosen as a convex combination of all the turning points found by ``cla_turning_points``. This will
yield a portfolio with the maximum Sharpe Ratio. The convex combination is found using the Golden section method.

**Solution String:** ``max_sharpe``

Minimum Variance
****************

The optimal weights will be chosen as a convex combination of all the turning points found by ``cla_turning_points``. This will yield a portfolio with minimum variance,

.. math::

      \sigma^{2} = w^T \sum w

where :math:`w` is the vector of weights, :math:`\sum` is the covariance matrix of elements in a portfolio.

**Solution String:** ``min_volatility``

Efficient Frontier
******************

Note that the turning points found by ``cla_turning_points`` constitute a small subset of all the points on the efficient frontier. This efficient frontier solution yields a list of all optimal weights lying on the efficient frontier and satisfying the problem. All these weights/points are found through the convex combination of CLA turning points.

**Solution String:** ``efficient_frontier``


Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.cla

    .. autoclass:: CriticalLineAlgorithm
        :members:

        .. automethod:: __init__

.. note::
    |h4| Using Custom Input |h4_|
    We provide great flexibility to the users in terms of the input data - they can either pass their own pre-calculated input
    matrices/dataframes or leave it to us to calculate them. A quick reference on common input parameters which you will encounter
    throughout the portfolio optimization module:

        * :py:mod:`asset_prices`: Dataframe/matrix of historical raw asset prices **indexed by date**.
        * :py:mod:`asset_returns`: Dataframe/matrix of historical asset returns. This will be a :math:`TxN` matrix where :math:`T` is the time-series and :math:`N` refers to the number of assets in the portfolio.
        * :py:mod:`expected_asset_returns`: List of expected returns per asset i.e. the mean of historical asset returns. This refers to the parameter :math:`\mu` used in portfolio optimization literature. For a portfolio of 5 assets, ``expected_asset_returns = [0.45, 0.56, 0.89, 1.34, 2.4]``.
        * :py:mod:`covariance_matrix`: The covariance matrix of asset returns.

.. tip::
    |h4| Specifying Weight Bounds |h4_|
    Users can specify weight bounds in two ways:

    * Use the same set of lower and upper bound values for all the assets simultaneously. In this case just pass a tuple of bounds.

        .. code::

            cla = CriticalLineAlgorithm(weight_bounds=(0.3, 1))

    * If you want to specify individual bounds, you need to pass a tuple of lists - the first list containing lower bounds and the second list containing upper bounds for all assets respectively. Something like this:

        .. code::

            cla = CriticalLineAlgorithm(weight_bounds=([0.2, 0.1, 0, 0, 0], [1, 1, 1, 0.9, 0.8]))

      Note that when using this way of passing bounds, you need to specify bounds for all the assets. For free assets i.e. those       with no specific bounds, just specify 0 as lower bound value and 1 as upper bound value.

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.cla import CriticalLineAlgorithm

    # Read in data
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute different solutions using CLA
    cla = CriticalLineAlgorithm()

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


Research Notebooks
##################

The following research notebooks provide a more detailed exploration of the algorithm.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Machine%20Learning%20Asset%20Allocation/Chapter16.ipynb