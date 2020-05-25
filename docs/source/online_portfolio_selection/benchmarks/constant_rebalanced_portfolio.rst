.. _online_portfolio_selection-benchmarks-constant_rebalanced_portfolio:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

=============================
Constant Rebalanced Portfolio
=============================

Constant Rebalanced Portfolio rebalances to a certain portfolio weight every time period. This particular weight can be set by the user,
and if there are no inputs, it will automatically allocate equal weights to all assets. The total returns for a CRP can be calculated by
taking the cumulative product of the weight and relative returns matrix.

.. math::
    S_t(CRP(b)) = \overset{t}{\underset{n=1}{\prod}} \:  b^{\top}x_n

Once the initial portfolio has been determined, the final weights can be represented as buying and holding the initial weight.

.. math::
    S_t(BEST) = \underset{b \in \Delta_m}{\max} b \cdot \left(\overset{t}{\underset{n=1}{\bigodot}}  x_n \right) = S_t(BAH(b_0))

- :math:`S(t)` is the total portfolio returns at time :math:`t`.
- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p(t)` is the price at time :math:`t`.
- :math:`\bigodot` is the element-wise cumulative product. In this case, the cumulative product represents the overall change in prices.
- :math:`\prod` is the product of all elements.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb>`_
    provides a more detailed exploration of the strategies.

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.constant_rebalanced_portfolio

    .. autoclass:: CRP
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

    # Compute Constant Rebalanced Portfolio weights with no weights given.
    crp = CRP()
    crp.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Constant Rebalanced Portfolio weights with given weights.
    crp = CRP()
    crp.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    crp.weights

    # Get all weights for the strategy.
    crp.all_weights

    # Get portfolio returns.
    crp.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_
