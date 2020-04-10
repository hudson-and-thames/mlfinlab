.. _portfolio_optimisation-risk_estimators:


===============
Risk Estimators
===============
This class includes functions for calculating different types of covariance matrices, de-noising, and other helpful methods.


Minimum Covariance Determinant
##############################

Minimum Covariance Determinant (MCD) is a robust robust estimator of covariance.

Our method is a wrapper for the sklearn MinCovDet class. For more details about the function and its parameters, please
visit `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`__.

Maximum Likelihood Covariance Estimator (Empirical Covariance)
##############################################################

Maximum Likelihood Estimator of a sample is an unbiased estimator of the corresponding population’s covariance matrix.

Our method is a wrapper for the sklearn EmpiricalCovariance class. For more details about the function and its parameters,
please visit `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html>`__.


Covariance Estimator with Shrinkage
###################################

Shrinkage allows one to avoid the inability to invert the covariance matrix due to numerical reasons. Shrinkage consists
of reducing the ratio between the smallest and the largest eigenvalues of the empirical covariance matrix.

The following Shrinkage methods are supported:

- Basic shrinkage
- Ledoit-Wolf shrinkage
- Oracle Approximating shrinkage

Our methods here are wrappers for the sklearn ShrunkCovariance, LedoitWolf, and OAS classes.
For more details about the function and its parameters, please visit `sklearn documentation <https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance>`__.

Semi-Covariance Matrix
#######################

Semi-Covariance matrix is used to measure the downside volatility of portfolio and can be used as a measure to minimize it.
This metric also allows to measure the volatility of returns below a specific threshold. An example of Semi-Covaraicne usage
can be found `here <https://www.solactive.com/wp-content/uploads/2018/04/Solactive_Minimum-Downside-Volatility-Indices.pdf>`__.

Exponentially-Weighted Covariance Matrix
########################################

Each element in the Exponentially-weighted Covariance matrix is the last element from exponentially weighted moving average
series based on series of covariances between returns of the corresponding assets. It's used to give greater weight to most
relevant observations in computing the covariance.

.. tip::
    This and above methods are described in more detail in the Risk Estimators Notebook.


De-noising Covariance Matrix
#############################

The main idea behind de-noising is to separate the noise-related eigenvalues from the signal-related ones. This is achieved
through fitting the Marcenko-Pastur distribution of the empirical distribution of eigenvalues using a Kernel Density Estimate (KDE).

.. tip::
    This method is described in more detail in the NCO Notebook.

Transforming Covariance to Correlation and Back
#############################################################

Helper methods for transforming the covariance matrix to the correlation matrix and back to the covariance matrix.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.risk_estimators

    .. autoclass:: RiskEstimators
        :members:

        .. automethod:: __init__

Research Notebooks
==================

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Risk Estimators Notebook`_

.. _Risk Estimators Notebook: https://github.com/hudson-and-thames/research/blob/master/RiskEstimators/RiskEstimators.ipynb

