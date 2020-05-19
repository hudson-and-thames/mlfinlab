.. _implementations-codependence_marti:

.. note::
   The following implementations and documentation, closely follows the work of Gautier Marti:
   `Some contributions to the clustering of financial time series and applications to credit default swaps <https://www.researchgate.net/publication/322714557>`_.

=====================
Codependence by Marti
=====================

The work mentioned above introduces a new approach of representing the random variables that splits apart dependency and
distribution without loosing any information. It also contains a distance metric between two financial time series based
on the novel approach.

According to the author's classification:

"Many statistical distances exist to measure the dissimilarity of two random variables, and therefore two i.i.d. random
processes. Such distances can be roughly classified in two families:

1. distributional distances, ... which focus on dissimilarity between probability distributions and quantify divergences
in marginal behaviours,

2. dependence distances, such as the distance correlation or copula-based kernel dependency measures ...,
which focus on the joint behaviours of random variables, generally ignoring their distribution properties.

However, we may want to be able to discriminate random variables both on distribution and dependence. This can be
motivated, for instance, from the study of financial assets returns: are two perfectly correlated random variables
(assets returns), but one being normally distributed and the other one following a heavy-tailed distribution, similar?
From risk perspective, the answer is no ..., hence the propounded distance of this article".

.. Tip::
   Read the original work to understand the motivation behind creating the novel technique deeper and check the reference
   papers that prove the above statements.

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


Implementation
==============

.. autofunction:: gpr_distance

Generic Non-Parametric Representation (GNPR) distance
#####################################################




Implementation
==============

.. autofunction:: gnpr_distance

Example
*******

The following example shows how the above functions can be used:

.. code-block::

   import pandas as pd
   from mlfinlab.codependence.gnpr_distance import spearmans_rho, gpr_distance, gnpr_distance

   # Getting the dataframe with time series of returns
   data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])
   element_x = 'SPY'
   element_y = 'TLT'

   # Calculating the Spearman's rho coefficient between two time series
   rho = spearmans_rho(data[element_x], data[element_y])

   # Calculating the GPR distance between two time series with both
   # distribution and dependence information
   gpr_dist = gpr_distance(data[element_x], data[element_y], theta=0.5)

   # Calculating the GNPR distance between two time series with dependence information only
   gnpr_dist = gnpr_distance(data[element_x], data[element_y], theta=1)
