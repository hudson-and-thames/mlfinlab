.. _statistical_arbitrage-introduction:
.. note::

    References

    1. `Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G., 2006. Pairs trading: Performance of a
    relative-value arbitrage rule. The Review of Financial Studies, 19(3), pp.797-827.
    <https://academic.oup.com/rfs/article/19/3/797/1646694>`_

=============
Pairs Trading
=============

Pairs trading strategy is a specific statistical arbitrage strategy that focuses on two assets.
Instead of trading on a basket of assets, pairs trading focuses on two to harness the pricing
inefficiency caused by the widening spread. Pairs trading strategies can be implemented in three parts.

1. Filter the universe to select a number of pairs. These pairs are two related securities,
which are oftentimes in the same sector/industry and have similar fundamental values.

2. Calculate the spread between the two pairs and test for stationarity and cointegration.

3. If all the tests are satisfied, generate trading signals to long the asset that is underpriced
and short the other.
