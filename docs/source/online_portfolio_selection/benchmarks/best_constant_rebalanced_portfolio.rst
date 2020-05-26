.. _online_portfolio_selection-benchmarks-best_constant_rebalanced_portfolio:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

==================================
Best Constant Rebalanced Portfolio
==================================

Best Constant Rebalanced Portfolio is a strategy that is implemented in hindsight, which is similar to Best Stock. It uses the same weight
for all time periods. However, it determines those weights by having the complete market sequence of the past. The objective function for
BCRP aims to maximize portfolio returns with the equation below.

.. math::
    b^{\bf{\star}} = \underset{b_t \in \Delta_m}{\arg\max} \: S_t(CRP(b)) = \underset{b \in \Delta_m}{\arg\max} \overset{t}{\underset{n=1}{\prod}} \:  b^{\top}x_n

Once the optimal weight has been determined, the final returns can be calculated by using the CRP returns equation.

.. math::
    S_t(BCRP) = \underset{b \in \Delta_m}{\max} \: S_t(CRP(b)) = S_t(CRP(b^{\bf \star}))

- :math:`S(t)` is the total portfolio returns at time :math:`t`.
- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`\bigodot` is the element-wise cumulative product. In this case, the cumulative product represents the overall change in prices.
- :math:`\prod` is the product of all elements.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb>`_
    provides a more detailed exploration of the strategies.

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.best_constant_rebalanced_portfolio

    .. autoclass:: BCRP
        :members:
        :show-inheritance:
        :inherited-members:

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection import *

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Best Constant Rebalanced Portfolio weights with no weights given.
    bcrp = BCRP()
    bcrp.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Get the latest predicted weights.
    bcrp.weights

    # Get all weights for the strategy.
    bcrp.all_weights

    # Get portfolio returns.
    bcrp.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_
