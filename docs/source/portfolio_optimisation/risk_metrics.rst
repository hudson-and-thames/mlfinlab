.. _portfolio_optimisation-risk_metrics:


============
Risk Metrics
============

The RiskMetrics class contains functions for calculation of common risk metrics used by investment professionals.
The list of supported metrics will grow with future updates of the package. Now, the following risk calculations are supported:

1. ``Variance``
2. ``Value at Risk (VaR)``
3. ``Expected Shortfall (CVaR)``
4. ``Conditional Drawdown at Risk (CDaR)``

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Portfolio Optimization with Drawdown Constraints** *by* Alexei Chekhlov, Stanislav Uryasev, Michael Zabarankin `available here <https://www.ise.ufl.edu/uryasev/files/2011/11/drawdown.pdf>`_. *Introduces a CDaR measure and compares it to the CVaR measure.*

Variance
########

This measure can be used to compare portfolios based on estimations of the volatility of returns.

The Variance of a portfolio is calculated as follows:

.. math::

      Var = w^{T} * Cov * w

Where :math:`w` is the vector of weights for instruments in a portfolio, and
:math:`Cov` is a covariance matrix of instruments in a portfolio. Result :math:`Var` is a scalar.

Value at Risk (VaR)
###################

This measure can be used to compare portfolios based on the amount of investments that can be lost in the next observation,
assuming the returns for assets follow a multivariate normal distribution.

The Value at Risk of a portfolio is calculated as follows:

.. math::

      VaR = Quantile_{\alpha}(R)

Where :math:`\alpha` is the confidence level to use, and the :math:`R` is a set of returns of a portfolio.

VaR of :math:`0.15` at :math:`\alpha = 0.05` level means that with a :math:`5\%` probability the portfolio will
decrease by :math:`15\%` on the next observation.

Expected Shortfall (CVaR)
#########################

This measure can be used to compare portfolios based on the average amount of investments that can be lost in a
worst-case scenario, assuming the returns for assets follow a multivariate normal distribution.

The Expected Shortfall of a portfolio is calculated as follows:

.. math::

      CVaR = E[{R, R < Quantile_{\alpha}(R)}]

Where :math:`\alpha` is the confidence level to use, and the :math:`R` is a set of returns of a portfolio.

CVaR of :math:`0.15` at :math:`\alpha = 0.05` level means that for :math:`5\%` worst cases of returns observations
the average loss of the portfolio value is :math:`15\%` per observation.

This picture from Y. Vardanyan demonstrates the differences between the VaR and the CVaR concepts:

.. image:: portfolio_optimisation_images/var_cvar_concepts.png
   :scale: 100 %
   :align: center

Conditional Drawdown at Risk (CDaR)
###################################

This measure can be used to compare portfolios based on the average amount of a portfolio drawdown in a
worst-case scenario, assuming the drawdowns follow a normal distribution.

The Expected Shortfall of a portfolio is calculated as follows:

.. math::
      :nowrap:

      \begin{align*}
      DD_{t} = max_{0 \le \tau \le t}\{w_{\tau}\} - w_{t}
      \end{align*}

      \begin{align*}
      CDaR = E[{DD_{t}, DD_{t} > Quantile_{\alpha}(DD)}]
      \end{align*}

Where :math:`\alpha` is the confidence level to use, :math:`w_{t}` is the price of a portfolio at time :math:`t`,
and :math:`DD_{t}` is the maximum historical drawdown up to time :math:`t` .

CDaR of :math:`0.15` at :math:`\alpha = 0.05` level means that for :math:`5\%` worst cases of historical portfolio drawdowns,
the average drawdown is :math:`0.15` units in which the portfolio price is measured.

.. tip::

    This risk metric is described in more detail in the work **Portfolio Optimization with Drawdown Constraints** `available here <https://www.ise.ufl.edu/uryasev/files/2011/11/drawdown.pdf>`_.

.. tip::

    VaR, CVaR and CDaR metrics can also be used for individual assets.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.risk_metrics

    .. autoclass:: RiskMetrics
        :members:

        .. automethod:: __init__

Example
########
Below is an example of how to use the package functions to calculate risk metrics for a portfolio.

.. code-block::

    import pandas as pd
    from mlfinlab.labeling import RiskMetrics

    # Import dataframe of returns for assets in a portfolio
    assets_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Calculate empirical covariance of assets
    assets_cov = assets_returns.cov()

    # Set weights for assets in a portfolio
    weights = [0.1, 0.2, 0.05, 0.05, 0.2, 0.1, 0.1, 0.2]

    # Pick a confidence interval
    alpha = 0.05

    # Class that contains needed functions
    risk_met = RiskMetrics()

    # Calculate Variance
    Var = risk_met.calculate_variance(assets_cov, weights)

    # Calculate Value at Risk of the first asset
    VaR = risk_met.calculate_value_at_risk(assets_returns.iloc[:,0], alpha)

    # Calculate Expected Shortfall
    CVaR = risk_met.calculate_expected_shortfall(assets_returns.iloc[:,0], alpha)

    # Calculate Conditional Drawdown at Risk
    CDaR = risk_met.calculate_conditional_drawdown_risk(assets_returns.iloc[:,0], alpha)
