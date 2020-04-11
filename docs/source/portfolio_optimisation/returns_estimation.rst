.. _portfolio_optimisation-returns_estimation:


=====================
Estimation of Returns
=====================

Accurate estimation of historical asset returns is one of the most important aspects of portfolio optimisation. At the same, it is
also one of the most difficult to calculate since most of the times, estimated returns do not correctly reflect the true underlying
returns of a portfolio/asset. Given this, there is still significant research work being published dealing with novel methods to
estimate returns and we wanted to share some of these methods with the users of mlfinlab.

This class provides functions to estimate mean asset returns. Currently, it is still in active development and we
will keep adding new methods to it.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.returns_estimators

    .. autoclass:: ReturnsEstimation
        :members:

        .. automethod:: __init__
