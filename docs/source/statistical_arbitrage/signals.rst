.. _statistical_arbitrage-signals:

=======
Signals
=======

There are many methods to implement statistical arbitrage, and thorough investigation for each
procedure is needed before applying these tools to a real live trading environment.

Kalman Filtering
################

To Be Implemented.

Ornstein-Uhlenbeck Process
##########################

.. note::
    `Bertram, W.K., 2010. Analytic solutions for optimal statistical arbitrage trading. Physica A: Statistical mechanics and its applications, 389(11), pp.2234-2243.
    <http://www.stagirit.org/sites/default/files/articles/a_0340_ssrn-id1505073.pdf>`_

The Ornstein-Uhlenbeck process is a stochastic mean-reverting process with the following equation:

.. math::
    dX_t = \kappa(\mu − X_t)dt + \sigma dW_t

- :math:`X_t`: Residual from the spread.
- :math:`\kappa`: Rate of mean reversion.
- :math:`\mu`: Mean of the process.
- :math:`\sigma`: Variance or volatility of the process.
- :math:`W_t`: Wiener process or Brownian motion.

This can be changed into an :math:`AR(1)` model with the following properties:

.. math::
    X_{n+1} = a + b X_n + \zeta_{n+1}

- :math:`b = e^{-\kappa \Delta_t}`
- :math:`a = \mu(1 - b)`
- :math:`var(\zeta) = \sigma^2 \frac{1 - b^2}{2 \kappa}`

We will primarily use the OU-process to generate trading signals for statistical arbitrage.
The trading signals will be defined as:

.. math::
    s = X_t - \frac{E(X_t)}{var(X_t)} = \frac{\mu\sqrt{2\kappa}}{\sigma}


Hurst Exponent
##############

To be implemented.
