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

    * The documentation of cvxpy is better than that of scipy, escpecially the parts related to optimisation.
    * cvxpy's code is much more readable and easier to understand.
    * **Note that cvxpy only supports convex optimisation problems as opposed to scipy.optimise which can also tackle concave problems**. Although this might seem as a downside, cvxpy raises clear error notifications if the problem is not convex and the required conditions are not met. This is very important for us as it ensures the solvability of an objective function - if there is no error from cvxpy's side, the objective function is correct and is guaranteed to run till completion by the optimiser.

Supported Portfolio Allocation Solutions
########################################

The MVO class provide some common portfolio optimisation problems out-of-the-box. In this section we go over a quick overview of
these:

Inverse Variance
*****************

For this solution, the diagonal of the covariance matrix is used for weights allocation.

.. math::

      w_{i} = \frac{\sum^{-1}}{\sum_{j=1}^{N}(\sum_{j,j})^{-1}}

where :math:`w_{i}` is the weight allocated to the :math:`i^{th}` asset in a portfolio, :math:`\sum_{i,i}` is the :math:`i^{th}`
element on the main diagonal of the covariance matrix of elements in a portfolio and :math:`N` is the number of elements in a
portfolio.

**Solution String:** ``inverse_variance``

Minimum Volatility
**********************

For this solution, the objective is to generate a portfolio with the least variance. The following optimisation problem is
being solved.

.. math::

    \begin{equation*}
        \begin{aligned}
            & \underset{\mathbf{w}}{\text{minimise}} & & w^T\sum w \\
            & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
            &&& w_{j} \geq 0, j=1,..,N
        \end{aligned}
    \end{equation*}

**Solution String:** ``min_volatility``

Maximum Sharpe Ratio
**********************

For this solution, the objective is (as the name suggests) to maximise the Sharpe Ratio of your portfolio.

.. math::

    \begin{equation*}
        \begin{aligned}
            & \underset{\mathbf{w}}{\text{maximise}} & & \frac{\mu^{T}w - R_{f}}{(w^{T}\sum w)^{1/2}} \\
            & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
            &&& w_{j} \geq 0, j=1,..,N
        \end{aligned}
    \end{equation*}

A major problem with the above formulation is that the objective function is not convex and this presents a problem for cvxpy
which only accepts convex optimisation problems. As a result, the problem can be transformed into an equivalent one, but with
a convex quadratic objective function.

.. math::

    \begin{equation*}
        \begin{aligned}
            & \underset{\mathbf{w}}{\text{minimise}} & & y^T\sum y \\
            & \text{s.t.} & & (\mu^{T}w - R_{f})^{T}y = 1 \\
            &&& \sum_{j=1}^{N}y_{j} = \kappa, \\
            &&& \kappa \geq 0, \\
            &&& w_{j} = \frac{y_j}{\kappa}, j=1,..,N
        \end{aligned}
    \end{equation*}

where :math:`y` refer to the set of unscaled weights, :math:`\kappa` is the scaling factor and the other symbols refer to
their usual meanings.

**Solution String:** ``max_sharpe``

.. tip::
    |h4| Underlying Literature |h4_|
    The process of deriving this optimisation problem from the standard maximising Sharpe ratio problem is described
    in the notes `IEOR 4500 Maximizing the Sharpe ratio <http://people.stat.sc.edu/sshen/events/backtesting/reference/maximizing%20the%20sharpe%20ratio.pdf>`_  from Columbia University.

Efficient Risk
**********************

For this solution, the objective is to minimise risk given a target return value by the investor. Note that the risk value for
such a portfolio will not be the minimum, which is achieved by the minimum-variance solution. However, the optimiser will find
the set of weights which efficiently allocate risk constrained by the provided target return, hence the name "efficient risk".

.. math::

    \begin{equation*}
        \begin{aligned}
            & \underset{\mathbf{w}}{\text{min}} & & w^T\sum w \\
            & \text{s.t.} & & \mu^Tw = \mu_t\\
            &&& \sum_{j=1}^{n}w_{j} = 1 \\
            &&& w_{j} \geq 0, j=1,..,N \\
        \end{aligned}
    \end{equation*}

where :math:`\mu_t` is the target portfolio return set by the investor and the other symbols refer to their usual meanings.

**Solution String:** ``efficient_risk``

Efficient Return
**********************

For this solution, the objective is to maximise the portfolio return given a target risk value by the investor. This is very
similar to the *efficient_risk* solution. The optimiser will find the set of weights which efficiently try to maximise return
constrained by the provided target risk, hence the name "efficient return".

.. math::

    \begin{equation*}
        \begin{aligned}
            & \underset{\mathbf{w}}{\text{max}} & & \mu^Tw \\
            & \text{s.t.} & & w^T\sum w = \sigma^{2}_t\\
            &&& \sum_{j=1}^{n}w_{j} = 1 \\
            &&& w_{j} \geq 0, j=1,..,N \\
        \end{aligned}
    \end{equation*}

where :math:`\sigma^{2}_t` is the target portfolio risk set by the investor and the other symbols refer to their usual meanings.

**Solution String:** ``efficient_return``

Maximum Return - Minimum Volatility
********************************************

This is often referred to as *quadratic risk utility.*

.. math::

    \begin{equation*}
        \begin{aligned}
            & \underset{\mathbf{w}}{\text{min}} & &  \lambda * w^T\sum w - \mu^Tw\\
            & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
            &&& w_{j} \geq 0, j=1,..,N \\
        \end{aligned}
    \end{equation*}

**Solution String:** ``max_return_min_volatility``

Maximum Diversification
***********************

Maximum Decorrelation
**********************

Custom Objective Function
********************************************

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



Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the algorithm as outlined at the back of Ch16 in
Advances in Financial Machine Learning.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Machine%20Learning%20Asset%20Allocation/Chapter16.ipynb
