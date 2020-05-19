.. _implementations-codependence_marti:

.. note::
   The following implementations and documentation, closely follows the work of Gautier Marti:
   `Some contributions to the clustering of financial time series and applications to credit default swaps <https://www.researchgate.net/publication/322714557>`_.

=====================
Codependence by Marti
=====================



Spearman’s Rho
##############

Following the work of Marti:

"[The Pearson correlation coefficient] suffers from several drawbacks:
- it only measures linear relationship between two variables;
- it is not robust to noise
- it may be undefined if the distribution of one of these variables have infinite second moment.

More robust correlation coefficients are copula-based dependence measures such as Spearman’s rho":

.. math::
    \rho_{S}(X, Y) &= 12 E[F_{X}(X), F_{Y}(Y)] - 3 \\
    &= \rho(F_{X}(X), F_{Y}(Y))

"and its statistical estimate":

.. math::
    \hat{\rho}_{S}(X, Y) = 1 - \frac{6}{T(T^2-1)}\sum_{t=1}^{T}(X^{(t)}- Y^{(t)})^2

where :math:`X` and :math:`Y` are univariate random variables, :math:`F_{X}(X)` is the cumulative distribution
function of :math:`X` , :math:`X^{(t)}` is the :math:`t` -th sorted observation of :math:`X` , and :math:`T` is the
total number of observations.

Our method is a wrapper for the scipy spearmanr function. For more details about the function and its parameters,
please visit `scipy documentation <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html>`_.

Implementation
==============

.. py:currentmodule:: mlfinlab.codependence.gnpr_distance

.. autofunction:: spearmans_rho

Generic Parametric Representation (GPR) distance
################################################




Generic Non-Parametric Representation (GNPR) distance
#####################################################




Implementation
==============

.. py:currentmodule:: mlfinlab.codependence.gnpr_distance

.. autofunction::


Example
*******

.. code-block::

   import pandas as pd
   from mlfinlab.codependence.gnpr_distance import (spearmans_rho, gpr_distance, gnpr_distance)

   X = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])
