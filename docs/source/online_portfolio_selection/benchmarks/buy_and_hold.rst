.. _online_portfolio_selection-benchmarks-buy_and_hold:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

============
Buy and Hold
============

Buy and Hold is a strategy where an investor invests in an initial portfolio and never rebalances it. The portfolio weights, however, change
as time goes by because the underlying assets change in prices.

Returns for Buy and Hold can be calculated by multiplying the initial weight and the cumulative product of relative returns.

.. math::
    S_t(BAH(b_1)) = b_1 \cdot \left(\overset{t}{\underset{n=1}{\bigodot}} x_n\right)

- :math:`S(t)` is the total portfolio returns at time :math:`t`.
- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`\bigodot` is the element-wise cumulative product. In this case, the cumulative product represents the overall change in prices.

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb>`_
    provides a more detailed exploration of the strategies.

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.buy_and_hold

    .. autoclass:: BAH
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

    # Compute Buy and Hold with uniform weights as no weights are given.
    bah = BAH()
    bah.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Buy and Hold weights with user given weights.
    bah = BAH()
    bah.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    bah.weights

    # Get all weights for the strategy.
    bah.all_weights

    # Get portfolio returns.
    bah.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_
