.. _optimal_mean_reverting_strategies-introduction:

============
Introduction
============

Optimal Mean-Reverting Strategies
#################################

Various asset prices such as commodities,
volatility indices, foreign exchange rates, etc. are known to exhibit mean reversion. But the most popular method
to use such characteristic is to construct mean-reverting portfolio prices by simultaneously taking positions in two highly correlated or co-moving
assets - an approach that is more widely known as *pairs trading*. As creating a spread gives the opportunity for the
statistical arbitrage, it doesn't come as a surprise that working with mean-reverting portfolios is quite popular
with hedge fund managers and investors.

However, the problem of how to determine when to open or close the position still holds. Do you, as an
investor, need to enter the market immediately or wait for the future opportunity? When to liquidate the position
after making the first trade? All these questions lead us to the investigation of the optimal sequential timing
of trades.

In this module, we will be formalizing the optimal stopping problem for assets or portfolios that have mean-reverting
dynamics and providing the solutions based on three mean-reverting models:

* Ornstein-Uhlenbeck (OU)
* Exponential Ornstein-Uhlenbeck (XOU)
* Cox-Ingersoll-Ross (CIR)


Naturally, the module is divided into three submodules for approaches to creating an optimal mean-reverting
strategy: ``OrnsteinUhlenbeck``, ``ExponentialOrnsteinUhlenbeck`` and ``CoxIngersollRoss``.

.. note::
   We are solving the optimal stopping problem for a mean-reverting portfolio that is constructed by holding :math:`\alpha`
   shares of a risky asset :math:`S^{(1)}` and and shorting :math:`\beta` of another risky asset :math:`S^{(2)}`,
   yielding a portfolio value:

   .. math::
      X_t^{\alpha,\beta} = \alpha S^{(1)} - \beta S^{(2)}, t \geq 0

   More information regarding this problem can be found in the following publication:

   `Optimal Mean reversion Trading: Mathematical Analysis and Practical Applications by Tim Leung and Xin Li <https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_ (p. 16)


