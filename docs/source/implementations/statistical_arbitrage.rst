.. _statistical_arbitrage-introduction:
.. note::

    References

    1. `Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G., 2006. Pairs trading: Performance of a relative-value arbitrage rule. The Review of Financial Studies, 19(3), pp.797-827. <https://academic.oup.com/rfs/article/19/3/797/1646694>`_


=====================
Statistical Arbitrage
=====================

Introduction
============

Statistical Arbitrage exploits the pricing inefficiency between two groups of assets. It was
first developed and used in the mid-1980s by Nunzio Tartaglia’s quantitative group at Morgan
Stanley. Later, David Shaw, the founder of D.E Shaw & Co, left Morgan Stanley and started his
own quantitative trading firm in the late 1980s dealing mainly in pair trading.

The strategy can be explained in a two-step process. First, two baskets of assets that have
historically moved similarly are identified. Then, the spread between the two is carefully
measured to look for signals of divergence. If the spread becomes wider than the value suggested
by historical data, the trader longs the losing basket and shorts the winning one. As the
spread reverts back to the mean, the positions will gain in value. Statistical arbitrage
is a contrarian strategy to harness the mean-reverting behavior of the pair ratio and is
strongly reliant on cointegration and mean-reversion to exploit the mispricing of the assets.



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

Stationarity
============

Augmented Dickey-Fuller Test
****************************

Phillips-Perron Test
********************

Phillips-Ouliaris Test
**********************

Cointegration
=============

Engle-Granger Test
******************

Johansen Test
**************

Optimal Trading Strategy
========================

Kalman Filtering
****************

Ornstein-Uhlenbeck Process
**************************

Hurst Exponent
**************
