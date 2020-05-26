.. _online_portfolio_selection-pattern_matching-symmetric_correlation_driven_nonparametric_learning:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

===================================================
Symmetric Correlation Driven Nonparametric Learning
===================================================

Market symmetry is a concept that the markets have mirrored price movements. Increasing price trends represents a mirror of a decreasing trend.
This gives us an intuitional understanding that if the price movements are strongly negatively correlated, the optimal portfolio weights
should minimize the returns or the losses from those periods as it is most likely that the optimal portfolio weights would be the inverse.

Introduced recently in a Journal of Financial Data Science paper by Yang Wang and Dong Wang in 2019, SCORN identifies positively
correlated windows and negatively correlated windows.

The positiviely correlated windows are identified similar to the process for CORN.

.. math::
    C(x_t;w,\rho) = \lbrace x_j \vert R(X^{j-1}_{j-2},X^{t-1}_{t-w})  > \rho)

And the negatively correlated windows are identified as any period with a correlation value below the negative of the threshold.

.. math::
    C'(x_t;w,\rho) = \lbrace x_j \vert R(X^{j-1}_{j-2},X^{t-1}_{t-w})  < -\rho)

The strategy, therefore, maximizes the returns for periods that are considered similar and minimize the losses over periods that are considered the opposite.

.. math::
    b^{\bf{\star}}_t(w,\rho) = \underset{b \in \Delta_m}{\arg\max} \underset{x \in C(x_t;w,\rho)}{\sum}\log b^{\top}x - \underset{x \in C'(x_t;w,\rho)}{\sum}\log b^{\top}x

Two different variations of the SCORN strategies are implemented in the Online Portfolio Selection module.

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`\rho` is the correlation threshold.
- :math:`R` is the correlation coefficient.
- :math:`C_t` is the set of similar periods.
- :math:`C'_t` is the set of similar periods.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Pattern%20Matching.ipynb>`_
    provides a more detailed exploration of the strategies.

1. Symmetric Correlation Driven Nonparametric Learning (SCORN)
##############################################################

SCORN Parameters
----------------

The optimal :math:`\rho` for SCORN is between 0 and 0.2. Most cases :math:`\rho` would be 0 to indicate a
binary classification regarding the similarity sets; however, there are some instances where a value of 0.2
is more optimal. The optimal window value more or less varies with a tendency for a shorter value of 1 or 2.
Although, there are cases where a window of 21 had the highest returns.

.. image:: pattern_matching_images/nyse_scorn.png
   :width: 49 %

.. image:: pattern_matching_images/msci_scorn.png
   :width: 49 %

SCORN Implementation
--------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.symmetric_correlation_driven_nonparametric_learning

    .. autoclass:: SCORN
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__


2. Symmetric Correlation Driven Nonparametric Learning - K (SCORN-K)
####################################################################

.. tip::

    More detailed explanation about the top-k strategy is available with the documentation for CORN-K.

SCORN-K Parameters
------------------

In general, :math:`\rho` of 1 is sufficient as most of the time the ideal :math:`\rho` is 0. For cases with
datasets that have optimal SCORN of 0.2, :math:`\rho` should be increased to 3. Window values are also dependent
on each data, but in most cases, value of 2 was sufficient.

.. image:: pattern_matching_images/nyse_scornk.png
   :width: 49 %

.. image:: pattern_matching_images/msci_scornk.png
   :width: 49 %

SCORN-K Implementation
----------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.symmetric_correlation_driven_nonparametric_learning_k

    .. autoclass:: SCORNK
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

    # SCORN
    # Compute Symmetric Correlation Driven Nonparametric Learning with no given weights, window of 1, and rho of 0.3.
    scorn = SCORN(window=1, rho=0.3)
    scorn.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Symmetric Correlation Driven Nonparametric Learning with user given weights, window of 3, and rho of 0.5.
    scorn1 = SCORN(window=3, rho=0.5)
    scorn1.allocate(asset_prices=stock_prices, weights=some_weight)

    # SCORN-K
    # Compute Symmetric Correlation Driven Nonparametric Learning - K with no given weights, window range of 10, rho of 7, and k of 2.
    scornk = SCORNK(window=10, rho=7, k=2)
    scornk.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Symmetric Correlation Driven Nonparametric Learning - K with user given weights, window range of 5, rho of 3, and k of 1.
    scornk1 = SCORNK(window=5, rho=3, k=1)
    scornk1.allocate(asset_prices=stock_prices, weights=some_weight)

    # Recalculate k for scornk1 to save computational time of generating all experts.
    scornk1.recalculate_k(k=3)

    # Get the latest predicted weights.
    scorn.weights

    # Get all weights for the strategy.
    scornk.all_weights

    # Get portfolio returns.
    scorn.portfolio_return

    # Get each object of the generated experts.
    scornk1.experts

    # Get each experts parameters.
    scornk.expert_params

    # Get all expert's portfolio returns over time.
    scornk.expert_portfolio_returns

    # Get capital allocation weights.
    scornk1.weights_on_experts

.. tip::

    Strategies were implemented with modifications from `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_
