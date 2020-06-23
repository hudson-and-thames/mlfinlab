.. _statistical_arbitrage-signals:

.. note::

    Strategies were implemented with modifications from:

    1. `Avellaneda, M. and Lee, J.H., 2010. Statistical arbitrage in the US equities market. Quantitative Finance, 10(7), pp.761-782.
    <https://www.tandfonline.com/doi/pdf/10.1080/14697680903124632>`_

=======
Signals
=======

There are many methods to implement statistical arbitrage, and thorough investigation for each
procedure is needed before applying these tools to a real live trading environment.

Ornstein-Uhlenbeck Process
##########################

The Ornstein-Uhlenbeck process is a stochastic mean-reverting process with the following equation:

.. math::
    dX_t = \kappa(m − X_t)dt + \sigma dW_t

- :math:`X_t`: Residual from the spread.
- :math:`\kappa`: Rate of mean reversion.
- :math:`m`: Mean of the process.
- :math:`\sigma`: Variance or volatility of the process.
- :math:`W_t`: Wiener process or Brownian motion.

This can be changed into an :math:`AR(1)` model with the following properties:

.. math::
    X_{n+1} = a + b X_n + \zeta_{n+1}

- :math:`b = e^{-\kappa \Delta_t}`
- :math:`a = m(1 - b)`
- :math:`var(\zeta) = \sigma^2 \frac{1 - b^2}{2 \kappa}`

S-Score
*******

We will primarily use the OU-process to generate trading signals for statistical arbitrage.
The trading signals will be defined as:

.. math::
    s = \frac{X_t -  m}{var(X_t)} = \frac{\mu\sqrt{2\kappa}}{\sigma}

A larger absolute value of s-score indicates a larger deviation from the mean.

.. py:currentmodule:: mlfinlab.statistical_arbitrage.signals

.. autofunction:: calc_ou_process

Mean Reversion Time
*******************

Mean reversion time is calculated by taking the inverse of :math:`\kappa`. The time then equates to:

.. math:: \frac{1}{\kappa}

The higher the number, the longer it will take to revert back to its projected mean.

Z-Score
#######

Z-Score is also another popular method of measuring the distance of the measurement from the mean.
It has an underlying assumption that the distribution is Gaussian, which may not necessarily be
true for many of the financial data available to us. Nonetheless, it is often used as the simple
but powerful equation indicates the magnitude of the deviation from the mean.

.. math::
    z = \frac{x - \mu}{\sigma}

- :math:`x`: Data point
- :math:`\mu`: Mean
- :math:`\sigma`: Standard Deviation

.. py:currentmodule:: mlfinlab.statistical_arbitrage.signals

.. autofunction:: calc_zscore
