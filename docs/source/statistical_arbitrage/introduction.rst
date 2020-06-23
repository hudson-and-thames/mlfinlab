.. _statistical_arbitrage-introduction:
.. note::

    A comprehensive review of traditonal statistical arbitrage strategies that has been extremely
    helpful for the development of the module:

    `Krauss, C., 2017. Statistical arbitrage pairs trading strategies: Review and outlook. Journal
    of Economic Surveys, 31(2), pp.513-545. <https://onlinelibrary.wiley.com/doi/pdf/10.1111/joes.12153>`_


============
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
    \frac{dP_t}{P_t} = \alpha dt + \beta \frac{dQ_t}{Q_t} + dX_t

- :math:`P_t`: Price of the first group of assets.
- :math:`Q_t`: Price of the second group of assets.
- :math:`\alpha`: Drift term. For the most part, we will assume that this value is 0.
- :math:`\beta`: Regression coefficient between the change in returns.
- :math:`X_t`: Cointegration residual.

We can interpret this in two different ways.

1. If :math:`X_t` is a significant positive value, we will long 1 unit of :math:`P_t` and short
:math:`\beta` unit of :math:`Q_t`.

2. If :math:`X_t` is a significant negative value, we will short 1 unit of :math:`P_t` and long
:math:`\beta` unit of :math:`Q_t`.

:math:`X_t` will be described much more in detail in the section for :doc:`signals`.

We can, therefore, interpret statistical arbitrage as a contrarian strategy to harness the
mean-reverting behavior of the pair ratio to exploit the mispricing of the assets.

Strategies
##########

This module currently employs two different statistical arbitrage strategies.

1. :doc:`pairs_trading`
2. :doc:`eigenportfolio`

Filtering
#########

There are multiple ways to filter the initial data. For a pairs trading example, the number of pairs
grows quadratically with :math:`n`. The number of total pairs is:

.. math::
    \frac{n(n-1)}{2}

If we only have 10 assets that we want to test for, the total number of pairs is :math:`\frac{10 * 9}{2} = 45`.
However, once we start scanning for a universe of stocks with over 5000 options, the numbers quickly
add up. Therefore, it is important to have an effective method to test before we start the initial process.
The most commonly used filtering method is the cointegration test. Using the cointegration test, we
can see which pairs of assets pass the threshold to reject the null hypothesis. More information is
available with :doc:`cointegration`.

For various measures of testing for codependency try different tests described from :doc:`../codependence/introduction`
