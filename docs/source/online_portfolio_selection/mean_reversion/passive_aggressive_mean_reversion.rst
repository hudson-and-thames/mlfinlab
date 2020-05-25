.. _online_portfolio_selection-mean_reversion-passive_aggressive_mean_reversion:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

=================================
Passive Aggressive Mean Reversion
=================================

Passive Aggressive Mean Reversion alternates between a passive and aggressive approach to the current market conditions.
The strategy can effectively prevent a huge loss and maximize returns by setting a threshold for mean reversion.

PAMR takes in a variable :math:`\epsilon`, a threshold for the market condition. If the portfolio returns for the period are
below :math:`\epsilon`, then PAMR will passively keep the previous portfolio, whereas if the returns are above the threshold,
the portfolio will actively rebalance to the less performing assets.

In a way, :math:`\epsilon` can be interpreted as the maximum loss for the portfolio. It is most likely that the asset that
decreased in prices for the period will bounce back, but there are cases where some companies plummet in value. PAMR
is an effective algorithm that will prevent huge losses in blindly following these assets.

PAMR defines a loss function:

.. math::
    l_{\epsilon} (b; x_t)

If the returns for the period are below the threshold, :math:`\epsilon`:

.. math::
    l_{\epsilon} (b; x_t) = 0

For returns that are higher than :math:`\epsilon`:

.. math::
    l_{\epsilon} (b; x_t) = b \cdot x_t - \epsilon

Typically :math:`\epsilon` is set at a value between 0 and 1 and closer to 1 as daily returns fluctuate around 1.

We will introduce three versions of Passive Aggressive Mean Reversion: PAMR, PAMR-1, and PAMR-2.

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p(t)` is the price at time :math:`t`.
- :math:`\epsilon` is the mean reversion threshold constant.

1. PAMR
#######

The first method is described as the following optimization problem:

.. math::
    b_{t+1} = \underset{b \in \Delta_m}{\arg\min} \frac{1}{2} \|b-b_t \|^2 \: \text{s.t.} \: l_{\epsilon}(b;x_t)=0

With the original problem formulation and :math:`\epsilon` parameters, PAMR is the most basic implementation.

2. PAMR-1
#########

PAMR-1 introduces a slack variable to PAMR.

:math:`C` is a positive parameter that can be interpreted as the aggressiveness of the strategy.

.. math::
    b_{t+1} = \underset{b \in \Delta_m}{\arg\min} \left\lbrace\frac{1}{2} \|b-b_t \|^2 + C\xi\right\rbrace \: \text{s.t.} \: l_{\epsilon}(b;x_t) \leq \xi \geq 0

A higher :math:`C` value indicates the affinity to a more aggressive approach.

2. PAMR-2
#########

PAMR-2 contains a quadratic term to the original slack variable from PAMR-1.

.. math::
    b_{t+1} = \underset{b \in \Delta_m}{\arg\min} \left\lbrace\frac{1}{2} \|b-b_t \|^2 + C\xi^2 \right\rbrace \: \text{s.t.} \: l_{\epsilon}(b;x_t) \leq \xi

By increasing the slack variable at a quadratic rate, the method regularizes portfolio deviations.

- :math:`C` is the aggressiveness of the strategy.
- :math:`\xi` is the slack variable used to calculate the optimization equation.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Mean%20Reversion.ipynb>`_
    provides a more detailed exploration of the strategies.

Parameters
----------

The optimal parameters depend on each dataset. For NYSE, aggressiveness was not an important parameter as
returns were primarily affected by the :math:`\epsilon` value. :math:`\epsilon` of 0 resulted as the
highest returns.

.. image:: mean_reversion_images/nyse_pamr.png
   :width: 32 %

.. image:: mean_reversion_images/nyse_pamr1.png
   :width: 32 %

.. image:: mean_reversion_images/nyse_pamr2.png
   :width: 32 %

For the US Equity dataset, the optimal :math:`\epsilon` is actually 1; however, the difference in returns,
for :math:`\epsilon` of 0 and 1 is not too far apart. Aggressiveness continues to be a nominal factor as
a hyperparameter.

.. image:: mean_reversion_images/equity_pamr.png
   :width: 32 %

.. image:: mean_reversion_images/equity_pamr1.png
   :width: 32 %

.. image:: mean_reversion_images/equity_pamr2.png
   :width: 32 %

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.mean_reversion.passive_aggressive_mean_reversion

    .. autoclass:: PAMR
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

    # Compute Passive Aggressive Mean Reversion with no given weights and epsilon of 0.3.
    pamr = PAMR(optimization_method=0, epsilon=0.3)
    pamr.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Passive Aggressive Mean Reversion - 1 with given user weights and epsilon of 0.4.
    pamr1 = PAMR(optimization_method=1, epsilon=0.4, agg=20)
    pamr1.allocate(asset_prices=stock_prices, weights=some_weight)

    # Compute Passive Aggressive Mean Reversion - 2 with given user weights and epsilon of 0.4.
    pamr2 = PAMR(optimization_method=2, epsilon=0.8, agg=1000)
    pamr2.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    pamr.weights

    # Get all weights for the strategy.
    pamr1.all_weights

    # Get portfolio returns.
    pamr2.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Zhao, P., Hoi, S.C., & Gopalkrishnan, V. (2012). PAMR: Passive aggressive mean
    reversion strategy for portfolio selection. Machine Learning, 87, 221-258.
    <https://link.springer.com/content/pdf/10.1007%2Fs10994-012-5281-z.pdf>`_

