.. _statistical_arbitrage-introduction:
.. note::

    References

    1. `Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G., 2006. Pairs trading: Performance of a
    relative-value arbitrage rule. The Review of Financial Studies, 19(3), pp.797-827.
    <https://academic.oup.com/rfs/article/19/3/797/1646694>`_


=====================
Statistical Arbitrage
=====================

Introduction
============

Statistical Arbitrage exploits the pricing inefficiency between two groups of assets. First
developed and used in the mid-1980s by Nunzio Tartaglia’s quantitative group at Morgan Stanley,
the classical strategy utilizes systematic trading signals and a market-neutral approach to
generate positive returns.

The strategy can be explained in a two-step process. First, two baskets of assets that have
historically moved similarly are identified. Then, the spread between the two is carefully
measured to look for signals of divergence. If the spread becomes wider than the value suggested
by historical data, the trader longs the losing basket and shorts the winning one. As the spread
reverts back to the mean, the positions will gain in value.

Most strategies involving statistical arbitrage can be expressed with the following equation:

.. math::
    \frac{dP_t}{P_t} = /alpha dt + /beta dQ_t Q_t + dX_t

- P_t: Price of the first group of assets.
- Q_t: Price of the second group of assets.
- \alpha: For the most parts, we will assume that this value is 0.
- \beta: Regression coefficient between the change in returns.
- \X_t: Cointegration residual.

This can be interpreted as going long 1 unit of :math:`P_t` and short :math:`/beta` unit of
:math:`Q_t` if :math:`X_t` is a significant positive value and vice versa for a significant
negative value of :math:`X_t`. Here we assume that :math:`X_t` is a stationary process with
mean-reverting tendencies. :math:`X_t` will be described much more in detail in the section
that describes the Ornstein-Uhlenbeck process.

We can, therefore, interpret statistical arbitrage as a contrarian strategy to harness the
mean-reverting behavior of the pair ratio to exploit the mispricing of the assets.

Pairs Trading
=============

Pairs trading strategies can be implemented in three parts.

1. Filter the universe to select a number of pairs. These pairs are two related securities,
which are oftentimes in the same sector/industry and have similar fundamental values.

2. Calculate the spread between the two pairs and test for stationarity and cointegration.

3. If all the tests are satisfied, generate trading signals to long the asset that is underpriced
and short the other.

Filtering
=========

There are multiple ways to filter the data. For a pairs trading example, the number of pairs grows
quadratically with :math:`n`. The number of total pairs is

.. math::
    \frac{n(n-1)}{2}

If we only have 10 assets that we want to test for, the total number of pairs is 45. However, once
we start scanning for a universe of stocks with over 5000 options, the numbers quickly add up.
Therefore, it is important to have an effective method to test before we start the initial process.
The filtering method that will be employed for this module will be the cointegration test. Using the
cointegration test, we will see which pairs of assets pass the threshold to reject the null hypothesis.
More information on *Cointegration* is available two headings below.

Not implemented in the module yet, but other options for filtering include:

1. Principal Component Analysis

    - Transforms data matrix to a set of principal components to reduce the dimensions.
    - `Avellaneda, M. and Lee, J.H., 2010. Statistical arbitrage in the US equities market. Quantitative Finance, 10(7), pp.761-782. <https://www.tandfonline.com/doi/pdf/10.1080/14697680903124632>`_

2. Clustering

    - Fundamental values
    - K-means

3. Heuristics

4. Distance

Stationarity
============

A time series is defined to be stationary if its joint probability distribution is invariant
under translations in time or space. In other words, the mean and variance of the time series
do not change.

It is important to test for the spread for stationarity as statistical arbitrage typically
shows the strongest and most robust results that follow stationarity and cointegration for
the tested pairs.

Augmented Dickey-Fuller Test
****************************

Augmented Dickey-Fuller or the ADF tests the null hypothesis that a unit root is present
in a time series sample. If the time series does have a mean-reverting trend, then the next
price will be proportional to the current. The original Dickey-Fuller test only tested for
lag 1, whereas the augmented version can test for lag up to :math:`p`.

.. math::
	\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + \cdots + \delta_{p-1} \Delta y_{t-p+1} + \epsilon_t

- :math:`\alpha`: constant variable
- :math:`\beta`: coefficient of temporal trend
- :math:`\delta`: change of :math`y`

For the purpose of this module, we will empirically set :math:`p` to be :math:`1`.

.. py:currentmodule:: mlfinlab.statistical_arbitrage.stationarity

.. autofunction:: calc_stationarity

Phillips-Perron Test
********************

To Be Implemented.

Phillips-Ouliaris Test
**********************

To Be Implemented.

Cointegration
=============

Engle-Granger Test
******************

.. py:currentmodule:: mlfinlab.statistical_arbitrage.cointegration

.. autofunction:: calc_cointegration

Johansen Test
**************

To Be Implemented.

Optimal Trading Strategy
========================

To Be Implemented.

Kalman Filtering
****************

To Be Implemented.

Ornstein-Uhlenbeck Process
**************************

To Be Implemented.

Hurst Exponent
**************

To Be Implemented.

Labelling
=========

Optimal Trading Rules
*********************


Optimal Portfolio Allocation
****************************
