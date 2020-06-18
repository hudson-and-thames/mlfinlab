.. _portfolio_optimisation-mean_variance:

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
    The portfolio optimization module contains different algorithms that are used for asset allocation and optimising strategies.
    Each algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight
    allocations on the specific user data. This way, each implementation can be called in the same way and this makes it simple
    for users to use them.


==========================
Mean-Variance Optimisation
==========================

Traditionally, portfolio optimization is nothing more than a simple mathematical optimization problem, where your objective is to
achieve optimal portfolio allocation bounded by some constraints. It can be mathematically expressed as follows:

.. math::
        \begin{align*}
            & \underset{\mathbf{x}}{\text{max}} & & f(\mathbf{x}) \\
            & \text{s.t.} & & g(\mathbf{x}) \leq 0 \\
            &&& h(\mathbf{x}) = 0 \\
        \end{align*}

where :math:`x \in R^n` and :math:`h(x), g(x)` represent convex functions correlating to the equality and inequality constraints
respectively. Based on the mean-variance framework first developed by Harry Markowitz, a portfolio optimization problem can be
formulated as follows,

.. math::
        \begin{align*}
            & \underset{\mathbf{w}}{\text{min}} & & w^T\sum w \\
            & \text{s.t.} & & \sum_{i=1}^{n}w_{i} = 1 \\
            &&& \mu^Tw = \mu_t \\
        \end{align*}

where :math:`w` refers to the set of weights for the portfolio assets, :math:`\sum` is the covariance matrix of the assets,
:math:`\mu` is the expected asset returns and :math:`\mu_t` represents the target portfolio return of the investor. Note that this
represents a very basic (and a specific) use-case of portfolio allocation where the investor wants to minimse the portfolio risk
for a given target return. As the needs of an investor increase, the complexity of the problem also changes with different
objective functions and multitude of constraints governing the optimal set of weights.

The MeanVarianceOptimisation() class (MVO) provides a very flexible framework for many common portfolio allocation problems
encountered in practice. Users need to simply call a master function and by specifying the required parameters, the MVO class does
the hard work by utilising a quadratic optimiser and calculating the optimal set of weights based on the problem and constraints
specified.

.. note::
    |h4| The Quadratic Optimiser |h4_|
    Many mean-variance objective functions are typical quadratic optimization problems and can be solved by using a black-box
    quadratic optimiser. We use `cvxpy <https://www.cvxpy.org/index.html>`_ as our quadratic optimiser instead of the more
    frequently used `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_. This was a design choice for the
    following reasons:

    * The documentation of cvxpy is better than that of scipy, escpecially the parts related to optimization.
    * cvxpy's code is much more readable and easier to understand.
    * **Note that cvxpy only supports convex optimization problems as opposed to scipy.optimise which can also tackle concave problems**. Although this might seem as a downside, cvxpy raises clear error notifications if the problem is not convex and the required conditions are not met. This is very important for us as it ensures the solvability of an objective function - if there is no error from cvxpy's side, the objective function is correct and is guaranteed to run till completion by the optimiser.

Supported Portfolio Allocation Solutions
########################################

MlFinLab's :py:mod:`MeanVarianceOptimisation` class provide some common portfolio optimization problems out-of-the-box. In this section we go over a quick overview of
these:

Inverse Variance
****************

For this solution, the diagonal of the covariance matrix is used for weights allocation.

.. math::

      w_{i} = \frac{\sum^{-1}}{\sum_{j=1}^{N}(\sum_{j,j})^{-1}}

where :math:`w_{i}` is the weight allocated to the :math:`i^{th}` asset in a portfolio, :math:`\sum_{i,i}` is the :math:`i^{th}`
element on the main diagonal of the covariance matrix of elements in a portfolio and :math:`N` is the number of elements in a
portfolio.

**Solution String:** ``inverse_variance``

Minimum Volatility
******************

For this solution, the objective is to generate a portfolio with the least variance. The following optimization problem is
being solved.

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{minimise}} & & w^T\sum w \\
        & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N
    \end{align*}

**Solution String:** ``min_volatility``

Maximum Sharpe Ratio
********************

For this solution, the objective is (as the name suggests) to maximise the Sharpe Ratio of your portfolio.

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{maximise}} & & \frac{\mu^{T}w - R_{f}}{(w^{T}\sum w)^{1/2}} \\
        & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N
    \end{align*}

A major problem with the above formulation is that the objective function is not convex and this presents a problem for cvxpy
which only accepts convex optimization problems. As a result, the problem can be transformed into an equivalent one, but with
a convex quadratic objective function.

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{minimise}} & & y^T\sum y \\
        & \text{s.t.} & & (\mu^{T}w - R_{f})^{T}y = 1 \\
        &&& \sum_{j=1}^{N}y_{j} = \kappa, \\
        &&& \kappa \geq 0, \\
        &&& w_{j} = \frac{y_j}{\kappa}, j=1,..,N
    \end{align*}

where :math:`y` refer to the set of unscaled weights, :math:`\kappa` is the scaling factor and the other symbols refer to
their usual meanings.

**Solution String:** ``max_sharpe``

.. tip::
    |h4| Underlying Literature |h4_|
    The process of deriving this optimization problem from the standard maximising Sharpe ratio problem is described
    in the notes `IEOR 4500 Maximizing the Sharpe ratio <http://people.stat.sc.edu/sshen/events/backtesting/reference/maximizing%20the%20sharpe%20ratio.pdf>`_  from Columbia University.

Efficient Risk
**************

For this solution, the objective is to minimise risk given a target return value by the investor. Note that the risk value for
such a portfolio will not be the minimum, which is achieved by the minimum-variance solution. However, the optimiser will find
the set of weights which efficiently allocate risk constrained by the provided target return, hence the name "efficient risk".

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{min}} & & w^T\sum w \\
        & \text{s.t.} & & \mu^Tw = \mu_t\\
        &&& \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N \\
    \end{align*}

where :math:`\mu_t` is the target portfolio return set by the investor and the other symbols refer to their usual meanings.

**Solution String:** ``efficient_risk``

Efficient Return
****************

For this solution, the objective is to maximise the portfolio return given a target risk value by the investor. This is very
similar to the *efficient_risk* solution. The optimiser will find the set of weights which efficiently try to maximise return
constrained by the provided target risk, hence the name "efficient return".

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{max}} & & \mu^Tw \\
        & \text{s.t.} & & w^T\sum w = \sigma^{2}_t\\
        &&& \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N \\
    \end{align*}

where :math:`\sigma^{2}_t` is the target portfolio risk set by the investor and the other symbols refer to their usual meanings.

**Solution String:** ``efficient_return``

Maximum Return - Minimum Volatility
***********************************

This is often referred to as *quadratic risk utility.* The objective function consists of both the portfolio return and the risk.
Thus, minimising the objective relates to minimising the risk and correspondingly maximising the return. Here, :math:`\lambda` is
the risk-aversion parameter which models the amount of risk the user is willing to take. A higher value means the investor will
have high defense against risk at the expense of lower returns and keeping a lower value will place higher emphasis on maximising
returns, neglecting the risk associated with it.

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{min}} & &  \lambda * w^T\sum w - \mu^Tw\\
        & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N \\
    \end{align*}

**Solution String:** ``max_return_min_volatility``

Maximum Diversification
***********************

Maximum diversification portfolio tries to diversify the holdings across as many assets as possible. In the 2008 paper, `Toward Maximum Diversification <https://blog.thinknewfound.com/2018/12/maximizing-diversification/#easy-footnote-bottom-1-6608>`_, the diversification ratio, :math:`D`, of a portfolio, is defined as:

.. math::

        D = \frac{w^{T}\sigma}{\sqrt{w^{T}\sum w}}

where :math:`\sigma` is the vector of volatilities and :math:`\sum` is the covariance matrix. The term in the denominator is the
volatility of the portfolio and the term in the numerator is the weighted average volatility of the assets. More diversification
within a portfolio decreases the denominator and leads to a higher diversification ratio. The corresponding objective function and
the constraints are:

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{max}} & &  D\\
        & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N \\
    \end{align*}

**Solution String:** ``max_diversification``

.. tip::
    |h4| Underlying Literature |h4_|
    You can read more about maximum diversification portfolio in the following blog post on the website *Flirting with Models:* `Maximizing Diversification <https://blog.thinknewfound.com/2018/12/maximizing-diversification/>`_.

Maximum Decorrelation
*********************

For this solution, the objective is to minimise the correlation between the assets of a portfolio

.. math::

    \begin{align*}
        & \underset{\mathbf{w}}{\text{min}} & &  w^TA w\\
        & \text{s.t.} & & \sum_{j=1}^{n}w_{j} = 1 \\
        &&& w_{j} \geq 0, j=1,..,N \\
    \end{align*}

where :math:`A` is the correlation matrix of assets. The Maximum Decorrelation portfolio is closely related to Minimum Variance and Maximum Diversification, but applies to the case where an investor believes all assets have similar returns and volatility, but heterogeneous correlations. It is a Minimum Variance optimization that is performed on the correlation matrix rather than the covariance matrix, :math:`\sum`.

**Solution String:** ``max_decorrelation``

.. tip::
    |h4| Underlying Literature |h4_|
    You can read more on maximum decorrelation portfolio in the following blog post: `Max Decorrelation Portfolio <https://systematicedge.wordpress.com/2013/05/12/max-decorrelation-portfolio/>`_.

Creating a Custom Portfolio Allocation
######################################

For most of the users, the above solutions will be enough for their use-cases. However, we also provide a way for users to create
their custom portfolio problem. **This includes complete flexibility to specify the input, optimization variables, objective function and the corresponding constraints**. Let us go through the step-by-step process of formulating your own allocation problem:

Non-CVXPY Variables
*******************

The first step is to specify input variables not related to cvxpy (i.e. not defined as cvxpy variable objects, :py:mod:`cvxpy.Variable`). This can include anything ranging from raw asset prices data to historical returns to integer or string variables. All data types are supported - ``int``, ``float``, ``str``, ``Numpy matrices/lists``, ``Python lists``, ``Pandas dataframe``.

.. code-block::

    data = pd.read_csv('stock_prices.csv', parse_dates=True, index_col="Date")
    non_cvxpy_variables = {
            'asset_prices': data,
            'num_assets': data.shape[1],
            'covariance': data.cov(),
            'asset_names': data.columns,
            'expected_returns': ReturnsEstimation().calculate_mean_historical_returns(asset_prices=data, resample_by='W')
    }

In the above code example, we initialise a dataframe of historical stock prices and then define the dictionary containing all user
required input variables. **The key of the dictionary is the variable name and the value is the Pythonic value you want to assign that variable.**

CVXPY Variables
***************

The second step is to specify the cvxpy specific variables which are declared in the syntax required by cvxpy. You can include as
many new variables as you need by initialising a simple Python list with each declaration being a string. **Each of these variables should be a** :py:mod:`cvxpy.Variable` **object.**

.. code-block::

    cvxpy_variables = [
            'risk = cp.quad_form(weights, covariance)',
            'portfolio_return = cp.matmul(weights, expected_returns)'
    ]

Here, we are declaring two new cvxpy variables - :py:mod:`risk` and :py:mod:`portfolio_return`. Note that we are using non-cvxpy
variables - :py:mod:`covariance` and :py:mod:`expected_returns` - declared in the previous step to initialise the new ones.

.. note::
    |h4| Variable for Portfolio Weights |h4_|

    Internally, the code declares a Python variable - :py:mod:`weights` - for the final portfolio weights. We request you to use
    this same variable name whenever you want to include it in one of your custom variable declarations. Refer to the above code
    snippet for an example.

    |h4| Calling CVXPY |h4_|

    Internally, cvxpy is imported as follows:

    .. code-block::

        import cvxpy as cp

    For creating any cvxpy specific variables, you need to reference the library as :py:mod:`cp` otherwise the code will fail to
    run and give you an error.

Custom Objective Function
**************************

The third step is to specify the objective function for our portfolio optimization problem. You need to simply pass a string form
of the Python code for the objective function.

.. code-block::

    custom_obj = 'cp.Minimize(risk)'


Optimisation Constraints
************************

This is an optional step which requires you to specify the constraints for your optimization problem. Similar to how we specified
cvxpy variables, the constraints need to be specified as a Python list with each constraint being a string representation.

.. code-block::

    constraints = ['cp.sum(weights) == 1', 'weights >= 0', 'weights <= 1']

.. warning::
    Please note that cvxpy does not support strict inequality constraints - ``>`` or ``<`` - and will fail to solve your
    problem if you do specify one.

Bringing it all together, the code looks like this:

.. code-block::

    mvo = MeanVarianceOptimisation()
    custom_obj = 'cp.Minimize(risk)'
    constraints = ['cp.sum(weights) == 1', 'weights >= 0', 'weights <= 1']
    non_cvxpy_variables = {
        'num_assets': self.data.shape[1],
        'covariance': self.data.cov(),
        'expected_returns': ReturnsEstimation().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                  resample_by='W')
    }
    cvxpy_variables = [
        'risk = cp.quad_form(weights, covariance)',
        'portfolio_return = cp.matmul(weights, expected_returns)'
    ]
    mvo.allocate_custom_objective(non_cvxpy_variables=non_cvxpy_variables,
                                  cvxpy_variables=cvxpy_variables,
                                  objective_function=custom_obj,
                                  constraints=constraints)
    print(mvo.weights)

.. note::
    |h4| Some Important Miscellaneous Points |h4_|

    * The custom allocation feature still uses cvxpy as the quadratic optimiser. Hence, only convex objective functions are accepted since cvxpy currently does not support non-convex functions. We plan on adding support for non-linear and non-convex objective solutions soon!

    * The order of declaring variables also matters here. All non-cvxpy and cvxpy variables are initialised in a linear order, i.e. traversing dictionary from top to bottom and the list from left to right, hence you need to specify them in the order you want it to. For e.g. the following code is wrong and will give an error,

        .. code-block::

            cvxpy_variables = [
                'x = cp.quad_form(weights, y)',
                'y = cp.Variable(1)'
            ]

      The formula for :py:mod:`x` uses :py:mod:`y`, but due to the order of the list, :py:mod:`y` will be initialised after
      :py:mod:`x` and give an error.


.. warning::
    Although we have written extensive unittests, the custom allocation code is still in an experimental stage and you may
    encounter errors which we may have failed to incorporate. We request you to raise an issue `here <https://github.com/hudson-and-thames/mlfinlab/issues>`_ and we will promptly push a fix for it.

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
    throughout the portfolio optimization module:

        * :py:mod:`asset_prices`: Dataframe/matrix of historical raw asset prices **indexed by date**.
        * :py:mod:`asset_returns`: Dataframe/matrix of historical asset returns. This will be a :math:`TxN` matrix where :math:`T` is the time-series and :math:`N` refers to the number of assets in the portfolio.
        * :py:mod:`expected_asset_returns`: List of expected returns per asset i.e. the mean of historical asset returns. This refers to the parameter :math:`\mu` used in portfolio optimization literature. For a portfolio of 5 assets, ``expected_asset_returns = [0.45, 0.56, 0.89, 1.34, 2.4]``.
        * :py:mod:`covariance_matrix`: The covariance matrix of asset returns.


Plotting
########

``plot_efficient_frontier()`` : Plots the efficient frontier. You can specify the minimum and maximum return till which you want
the frontier to be displayed.

.. code-block::

    mvo = MeanVarianceOptimisation()
    expected_returns = ReturnsEstimation().calculate_mean_historical_returns(asset_prices=self.data,
                                                                             resample_by='W')
    covariance = ReturnsEstimation().calculate_returns(asset_prices=self.data, resample_by='W').cov()
    plot = mvo.plot_efficient_frontier(covariance=covariance,
                                       max_return=1.0,
                                       expected_asset_returns=expected_returns)

.. image:: portfolio_optimisation_images/efficient_frontier.png



Research Notebooks
##################

The following research notebooks provide a more detailed exploration of the algorithm.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Machine%20Learning%20Asset%20Allocation/Chapter16.ipynb