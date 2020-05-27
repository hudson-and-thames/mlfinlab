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

.. _portfolio_optimisation-mean_variance:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies.
    Each algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight
    allocations on the specific user data. This way, each implementation can be called in the same way and this makes it simple
    for users to use them.


==========================
Mean-Variance Optimisation
==========================

Traditionally, portfolio optimisation is nothing more than a simple mathematical optimisation problem, where your objective is to
achieve optimal portfolio allocation bounded by some constraints. It can be mathematically expressed as follows:

.. math::
        \begin{equation*}
            \begin{aligned}
                & \underset{\mathbf{x}}{\text{max}} & & f(\mathbf{x}) \\
                & \text{s.t.} & & g(\mathbf{x}) \leq 0 \\
                &&& h(\mathbf{x}) = 0 \\
            \end{aligned}
        \end{equation*}

where :math:`x \in R^n` and :math:`h(x), g(x)` represent convex functions correlating to the equality and inequality contraints
respectively. Based on the mean-variance framework first developed by Harry Markowitz, a portfolio optimisation problem can be
formulated as follows,

.. math::
        \begin{equation*}
            \begin{aligned}
                & \underset{\mathbf{w}}{\text{min}} & & w^T\sum w \\
                & \text{s.t.} & & \sum_{i=1}^{n}w_{i} = 1 \\
                &&& \mu^Tw = \mu_t \\
            \end{aligned}
        \end{equation*}
where :math:`w` refers to the set of weights for the portfolio assets, :math:`\sum` is the covariance matrix of the assets,
:math:`\mu` is the expected asset returns and :math:`\mu_t` represents the target portfolio return of the investor. Note that this
represents a very basic (and a specific) use-case of portfolio allocation where the investor wants to minimse the portfolio risk
for a given target return. As the needs of an investor increases, the complexity of the problem also changes with different
objective functions and multitude of constraints governing the optimal set of weights.

The MeanVarianceOptimisation() class (MVO) provides a very flexible framework for many common portfolio allocation problems
encountered in practice. Users need to simply call a master function and by specifying the required parameters, the MVO class does
the hard work by utilising a quadratic optimiser and calculating the optimal set of weights based on the problem and constraints
specified.

.. note::
    |h4| The Quadratic Optimiser |h4_|
    Many mean-variance objective functions are typical quadratic optimisation problems and can be solved by using a black-box
    quadratic optimiser. We use `cvxpy <https://www.cvxpy.org/index.html>`_ as our quadratic optimiser instead of the more
    frequently used `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_. This was a design choice for the
    following reasons:

    * the documentation of cvxpy is better than that of scipy.
    * cvxpy's code is much more readable and easier to understand.
    * scipy.optimise

Solutions
#########

The MVO class

    |h5| 1. Inverse Variance |h5_|

    With this solution string, only the main diagonal of the covariance matrix is used for weights allocation:

    .. math::

          W_{i} = \frac{\frac{1}{Cov_{i,i}}}{\sum_{j=1}^{N}{\frac{1}{Cov_{j,j}}}}

    Where :math:`W_{i}` is the weight allocated to the :math:`i` -th element in a portfolio, :math:`Cov_{i,i}` is the :math:`i`-th
    element on the main diagonal of the covariance matrix of elements in a portfolio, :math:`N` is the number of elements in a
    portfolio.

    |h5| 2. Minimum Volatility |h5_|

    With this solution string, the entire covariance matrix is used for weights allocation.

    The following optimisation problem is being solved:

    .. math::

          minimise: W^{T} * Cov * W

          s.t.: \sum_{j=1}^{N}{W_{j}} = 1

    Where :math:`W` is the vector of weights, :math:`Cov` is the covariance matrix of elements in a portfolio,
    :math:`N` is the number of elements in a portfolio.

    |h5| 3. Maximum Sharpe Ratio |h5_|

    With this solution string, the entire covariance matrix, the vector of mean returns, and the risk-free ratio are used
    for weights allocation.

    The standard problem of maximum Sharpe ratio portfolio optimization, formulated as:

    .. math::
          :nowrap:

          \begin{align*}
          maximise: \frac{MeanRet_{j} * W^{T} - R_{f}}{(W^{T} * Cov * W)^{1/2}}
          \end{align*}

          \begin{align*}
          s.t.: \sum_{j=1}^{N}{W_{j}} = 1
          \end{align*}

          \begin{align*}
          W_{j} \ge 0, j=1,..,N
          \end{align*}

    has the objective function being possibly non-concave. Therefore, it's not a convex optimization problem.

    However, the problem can be transformed into an equivalent one, but with a convex quadratic objective function:

    .. math::
          :nowrap:

          \begin{align*}
          minimise: Y^{T} * Cov * Y
          \end{align*}

          \begin{align*}
          s.t.: \sum_{j=1}^{N}{(MeanRet_{j} - R_{f}) * Y_{j}} = 1
          \end{align*}

          \begin{align*}
          \sum_{j=1}^{N}{Y_{j}} = \kappa
          \end{align*}

          \begin{align*}
          \kappa \ge 0
          \end{align*}

    After the optimisation: :math:`W_{j} = Y_{j} / \kappa`

    Where :math:`W` is the vector of weights, :math:`Y` is the vector of unscaled weights, :math:`\kappa` is the scaling factor,
    :math:`Cov` is the covariance matrix of elements in a portfolio, :math:`MeanRet` is the vector of mean returns,
    :math:`R_{f}` is the risk-free rate, :math:`N` is the number of elements in a portfolio.

    .. tip::

        The process of deriving this optimisation problem from the standard maximising Sharpe ratio problem is described
        in the notes `IEOR 4500 Maximizing the Sharpe ratio <http://people.stat.sc.edu/sshen/events/backtesting/reference/maximizing%20the%20sharpe%20ratio.pdf>`_  from Columbia University.

    |h5| 4. Efficient Risk |h5_|

    With this solution string, the entire covariance matrix, the vector of mean returns, and the target return are used
    for weights allocation.

    The following optimisation problem is being solved:

    .. math::
          :nowrap:

          \begin{align*}
          minimise : W^{T} * Cov * W
          \end{align*}

          \begin{align*}
          s.t.: \sum_{j=1}^{N}{MeanRet_{j} * W_{j}} = TrgetRet
          \end{align*}

          \begin{align*}
          \sum_{j=1}^{N}{W_{j}} = 1
          \end{align*}

    Where :math:`W` is the vector of weights, :math:`Cov` is the covariance matrix of elements in a portfolio,
    :math:`MeanRet` is the vector of mean returns, :math:`TrgetRet` is the target return, :math:`N` is the number of elements in a portfolio.

    .. tip::

        Note that users can also specify upper and lower bounds for asset weights:

        - Either a single upper and lower bound value can be applied for to all the asset weights in which case a single
          tuple needs to be passed: (low, high). By default a bound of (0, 1) is applied.
        - If individual bounds are required, then a dictionary needs to be passed with the key being the asset index and
          the value being the tuple of lower and higher bound values. Something like this: ``{asset_index : (low_i, high_i)}``

    |h5| 4. Maximum Return - Minimum Volatility |h5_|

    |h5| 5. Efficient Return |h5_|

    |h5| 6. Maximum Diversification |h5_|

    |h5| 7. Maximum Decorrelation |h5_|

    |h5| 8. Custom Objective Function |h5_|

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.mean_variance

    .. autoclass:: MeanVarianceOptimisation
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


Plotting
########

``plot_efficient_frontier()`` : Plots the efficient frontier. The red dot corresponds to the Maximum Sharpe portfolio.

.. code-block::::

    mvo = MeanVarianceOptimisation()
    mvo.allocate(asset_prices=stock_prices, resample_by='B')

    # Assuming there is a stock_returns dataframe
    mvo.plot_efficient_frontier(covariance=stock_returns.cov(),
                                expected_asset_returns=stock_returns.mean()*252,
                                num_assets=len(stock_returns.columns))

.. image:: portfolio_optimisation_images/efficient_frontier.png


Example Code
############

Basic example
*************

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation

    # Read in data
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute IVP weights
    mvo = MeanVarianceOptimisation()
    mvo.allocate(asset_names=stock_prices.columns, asset_prices=stock_prices,
                 solution='inverse_variance', resample_by='B
    ivp_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1)

.. note::

    We provide great flexibility to the users in terms of the input data - either they can pass raw historical stock prices
    as the parameter :py:mod:`asset_prices` in which case the expected returns and covariance matrix will be calculated
    using this data. Else, they can also pass pre-calculated :py:mod:`expected_returns` and :py:mod:`covariance_matrix`.

Different solutions
*******************

.. code-block::

    # Compute different mean-variance solutions using MVO
    mvo = MeanVarianceOptimisation()

    # Maximum Sharpe Solution
    mvo.allocate(asset_prices=stock_prices, solution='max_sharpe')
    mvo_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1)

    # Minimum Variance Solution
    mvo.allocate(asset_prices=stock_prices, solution='min_volatility')
    mvo_weights = mvo.weights.sort_values(by=0, ascending=False, axis=1)

    # Efficient Risk Solution
    mvo.allocate(asset_prices=stock_prices, solution='efficient_risk', target_return=0.4)
    mvo_weights = mvo.weights

    # Portfolio Characteristics
    portfolio_return = mvo.portfolio_return
    sharpe_ratio = mvo.portfolio_sharpe_ratio
    risk = mvo.portfolio_risk


Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the algorithm as outlined at the back of Ch16 in
Advances in Financial Machine Learning.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Machine%20Learning%20Asset%20Allocation/Chapter16.ipynb
