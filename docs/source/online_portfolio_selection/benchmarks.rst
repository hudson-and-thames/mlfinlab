.. _online_portfolio_selection-benchmarks:

.. note::

    Strategies were implemented with modifications from `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

==========
Benchmarks
==========

Before we dive into the more interesting and complex models of portfolio selection, we will begin our analysis with benchmarks.
As unappealing as benchmarks are, traditional strategies such as tracking the S&P 500 have been hugely successful.

Typically these are implemented in hindsight, so future data is often incorporated within the selection algorithm. For real-life
applications, we do not have access to future data from the present, so strategies here should be taken with a grain of salt.

There are four benchmarks strategies implemented in the Online Portfolio Selection module.

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

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.buy_and_hold

    .. autoclass:: BAH
        :members:
        :inherited-members:


Example Code
------------

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

Best Stock
==========

Best Stock strategy chooses the best performing asset in hindsight.

The best performing asset is determined with an argmax equation stated below. The portfolio selection strategy searches for the asset
that increases the most in price for the given time period.

.. math::
    b_0 = \underset{b \in \Delta_m}{\arg\max} \: b \cdot \left(\overset{n}{\underset{t=1}{\bigodot}}  x_t \right)

Once the initial portfolio has been determined, the final weights can be represented as buying and holding the initial weight.

.. math::
    S_t(BEST) = \underset{b \in \Delta_m}{\max} b \cdot \left(\overset{t}{\underset{n=1}{\bigodot}}  x_n \right) = S_t(BAH(b_0))

- :math:`S(t)` is the total portfolio returns at time :math:`t`.
- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`\bigodot` is the element-wise cumulative product. In this case, the cumulative product represents the overall change in prices.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.best_stock

    .. autoclass:: BestStock
        :members:
        :inherited-members:


Example Code
------------

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection import *

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Best Stock weights with no weights given.
    beststock = BestStock()
    beststock.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Get the latest predicted weights.
    beststock.weights

    # Get all weights for the strategy.
    beststock.all_weights

    # Get portfolio returns.
    beststock.portfolio_return

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
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
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
        :inherited-members:

        .. automethod:: __init__


Example Code
------------

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

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.best_constant_rebalanced_portfolio

    .. autoclass:: BCRP
        :members:
        :inherited-members:

Example Code
------------

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

Research Notebook
=================

The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb>`_
provides a more detailed exploration of the strategies.