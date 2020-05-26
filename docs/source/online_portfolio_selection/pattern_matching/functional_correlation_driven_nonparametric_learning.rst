.. _online_portfolio_selection-pattern_matching-functional_correlation_driven_nonparametric_learning:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

====================================================
Functional Correlation Driven Nonparametric Learning
====================================================

FCORN further extends the SCORN by introducing a concept of an activation function. Applying the concept
to the previous CORN algorithms, the activation function for the SCORN can be considered as a piecewise function.
For any value between the positive and negative value of the threshold, we discount the importance of the period by placing a constant of 0.

Instead of completely neglecting windows with correlation with absolute value less than the threshold,
FCORN introduces a sigmoid function that places a set of different weights depending on the correlation
to the current market window. By replacing with such a variable, it is possible for us to place different
importance on the correlated periods. One that has higher correlation will have higher weights of importance
whereas ones that are less correlated will have less importance on it.

The activation function can be labeled as the following:

.. math::
    b^{\bf{\star}}_t(w,\rho) = \underset{b \in \Delta_m}{\arg\max} \underset{j \in \lbrace1,...,t-1\rbrace}{\sum}v(j)\log b^{\top}x_i

If the correlation is nonnegative, we place a positive weight.

.. math::
    \text{if} \: c \geq 0 \rightarrow v(j) =  \frac{1}{1 + \exp(-\lambda(c-\rho))}

If the correlation is negative, we place a negative weight that approaches 0 for correlation values closer to 0.

.. math::
    \text{if} \: c < 0 \rightarrow v(j) =  \frac{1}{1 + \exp(-\lambda(c+\rho))} - 1

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`c` is the correlation coefficient.
- :math:`\rho` is the correlation threshold.
- :math:`v(j)` is the activation function for the given period weights.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

Two different variations of the FCORN strategies are implemented in the Online Portfolio Selection module.

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Pattern%20Matching.ipynb>`_
    provides a more detailed exploration of the strategies.

1. Functional Correlation Driven Nonparametric Learning (FCORN)
###############################################################

FCORN Parameters
----------------

The optimal :math:`\rho` for FCORN is between 0.4 and 0.8 with best lambd value at 1.
In most cases, window should be in the smaller range with 1 or 2 as seen with the case for the NYSE dataset; however,
SP500 has the highest returns with window of 5.

.. image:: pattern_matching_images/nyse_fcorn.png
   :width: 49 %

.. image:: pattern_matching_images/sp500_fcorn.png
   :width: 49 %

FCORN Implementation
--------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.functional_correlation_driven_nonparametric_learning

    .. autoclass:: FCORN
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__


2. Functional Correlation Driven Nonparametric Learning - K (FCORN-K)
#####################################################################

.. tip::

    More detailed explanation about the top-k strategy is available with the documentation for CORN-K.

FCORN-K Parameters
------------------

:math:`\rho` should be at least 5 to capture the range between 0.4 and 0.8, with lambd of 1 sufficient
to get the highest returns. Window values vary with each dataset, but a value of 1 or 2 typically
had the highest returns.

FCORN-K Implementation
----------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.functional_correlation_driven_nonparametric_learning_k

    .. autoclass:: FCORNK
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection import *

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # FCORN
    # Compute Functional Correlation Driven Nonparametric Learning with no given weights, window of 1, rho of 0.3, and lambd of 10.
    fcorn = FCORN(window=1, rho=0.3, lambd=10)
    fcorn.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Functional Correlation Driven Nonparametric Learning with user given weights, window of 3, rho of 0.5, and lambd of 5.
    fcorn1 = FCORN(window=3, rho=0.5, lambd=5)
    fcorn1.allocate(asset_prices=stock_prices, weights=some_weight)

    # FCORN-K
    # Compute Functional Correlation Driven Nonparametric Learning - K with no given weights, window of 10, rho of 7, lambd of 1, and k of 2.
    fcornk = FCORNK(window=10, rho=7, lambd=1, k=2)
    fcornk.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Functional Correlation Driven Nonparametric Learning - K with user given weights, window of 5, rho of 3, lambd of 2, and k of 1.
    fcornk1 = FCORNK(window=5, rho=3, lambd=2, k=1)
    fcornk1.allocate(asset_prices=stock_prices, weights=some_weight)

    # Recalculate k for fcornk1 to save computational time of generating all experts.
    fcornk1.recalculate_k(k=3)

    # Get the latest predicted weights.
    fcorn.weights

    # Get all weights for the strategy.
    fcornk.all_weights

    # Get portfolio returns.
    fcorn.portfolio_return

    # Get each object of the generated experts.
    fcornk1.experts

    # Get each experts parameters.
    fcornk.expert_params

    # Get all expert's portfolio returns over time.
    fcornk.expert_portfolio_returns

    # Get capital allocation weights.
    fcornk1.weights_on_experts

.. tip::

    Strategies were implemented with modifications from `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_
