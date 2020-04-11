.. _portfolio_optimisation-risk_metrics:


============
Risk Metrics
============

The RiskMetrics class contains functions for calculation of common risk metrics used by investment professionals. With time, we
will keep adding new metrics. For now, it supports the following risk calculations:

1. ``Variance``
2. ``Value at Risk (VaR)``
3. ``Expected Shortfall (CVaR)``
4. ``Conditional Drawdown at Risk (CDaR)``

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.risk_metrics

    .. autoclass:: RiskMetrics
        :members:

        .. automethod:: __init__
