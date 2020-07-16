.. _codependence-optimal_trnsport:

.. note::
   The following documentation closely follows the work by Marti et al.:
   `Exploring and measuring non-linear correlations: Copulas, Lightspeed Transportation and Clustering <https://arxiv.org/pdf/1610.09659.pdf>`_.

   Initial implementation was taken from the blog post by Gautier Marti:
   `Measuring non-linear dependence with Optimal Transport <https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html>`_.

=================
Optimal Transport
=================

The basic idea of the optimal copula transport dependence measure is rather simple. It relies on leveraging:

    1. Copulas, which are distributions encoding fully the dependence between random variables.

    2. A geometrical point of view: Where does the empirical copula stand in the space of copulas?
       In particular, how far is it from reference copulas such as the Fréchet–Hoeffding copula bounds
       (copulas associated to comonotone, countermonotone, independent random variables)?

.. figure:: images/optimal_transport_distance.png
   :scale: 100 %
   :align: center
   :figclass: align-center
   :alt: Optimal Transport distance

   Dependence is measured as the relative distance from independence to the nearest target-dependence: comonotonicity or countermonotonicity. (Source: Measuring non-linear dependence with Optimal Transport)

With this geometric view,

    1. It is rather easy to extend this novel dependence measure to alternative use cases
       (e.g. by changing the reference copulas).

    2. It can also allow to look for specific patterns in the dependence structure
       (generalization of conditional correlation): Typically, one would try to answer
       “What is the correlation between these two stocks knowing that the VIX is above 40?”
       (e.g. estimate ρXY|Z>z;

With this optimal copula transport tool, one can look for answers to, for example:

    a. “Which pairs of assets are both positively and negatively correlated?”

    b. “Which assets occur extreme variations while those of others are relatively small, and conversely?”

    c. “Which pairs of assets are positively correlated for small variations but uncorrelated otherwise?”

.. Tip::
   For an example comparing the behaviour of Optimal Transport distacne to Pearson's correlation, Spearman's rho,
   and Kendall tau distance, please read this blogpost by Gautier Marti:
   `Measuring non-linear dependence with Optimal Transport <https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html>`_.


Optimal Transport distance
==========================

According to the description of the method from the original paper by Marti et al.:

This methodology allows using copulas to encode dependence between variables, state-of-the-art optimal transport
for providing a relevant geometry to the copulas, andclustering for summarizing the main dependence patterns found
between the variables.

The idea of the approach is to "target specific dependence patterns and ignore others. We want to target dependence
which are relevant to such or such problem, and forget about the dependence which are not in the scope of the problems
at hand, or even worse which may be spurious associations (pure chance or artifacts in the data)."


The proposed codepependence coefficient "can be parameterized by a set of target-dependences, and a set
of forget-dependences. Sets of target and forget dependences can be built using expert hypotheses,
or by leveraging the centers of clusters resulting from an exploratory clustering of the pairwise dependences.
To achieve this goal, we will leverage three tools: copulas, optimal transportation, and clustering."


Optimal transport is based on the underlying theory of the Earth Mover’s Distance. Until very recently,
optimal transportation distances between distributions were not deemed relevant for machine learning applications
since the best computational cost known was super-cubic to the number of bins used for discretizing the
distribution supports which grows itself exponentially with the dimension. A mere distance evaluation
could take several seconds!

Copulas and optimal transportation are not yet mainstream tools, but they have recently gained attention
in machine learning, and several copula-based dependence measures have been proposed for improving
feature selection methods.


Copulas are functions that couple multivariate distribution functions to their univariate marginal distribution functions.

According to the original paper, bivariate copulas are considered, as higher dimensions would cost high computational burden.
But but most of the results and the methodology presented hold in the multivariate setting.


**Theorem 1 (Sklar’s Theorem)** Let :math:`X = (X_i, X_j)` be  a random  vector with  a joint
cumulative distribution function :math:`F` , and having continuous marginal cumulative
distribution functions :math:`F_i, F_j` respectively. Then, there exists a unique
distribution :math:`C` such that :math:`F(X_i, X_j) = C(F_i(X_i), F_j(X_j))` .
:math:`C` , the copula of :math:`X` , is the bivariate distribution of uniform
marginals :math:`U_i, U_j:=F_i(X_i), F_j(X_j)`

Implementation
##############

.. py:currentmodule:: mlfinlab.codependence.optimal_transport

.. autofunction:: optimal_transport_distance

Examples
========

The following example shows how the optimal transport distance can be used:

.. code-block::

   import pandas as pd
   from mlfinlab.codependence import optimal_transport_distance,
                              get_dependence_matrix

   # Getting the dataframe with time series of returns
   data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

   element_x = 'SPY'
   element_y = 'TLT'

   # Calculating the optimal transport distance between chosen assets
   # using Gaussian target copula with correlation 0.6
   ot_gaussian = optimal_transport_distance(data[element_x], data[element_y],
                                            target_dependence='gaussian',
                                            gaussian_corr=0.7)

   # Calculating the optimal transport distance between chosen assets
   # using comonotonicity target copula with correlation 0.6
   ot_comonotonicity = optimal_transport_distance(data[element_x], data[element_y],
                                                  target_dependence='comonotonicity',
                                                  gaussian_corr=0.7)

   # Calculating the optimal transport distance between all assets
   # using positive-negative target copula
   ot_matrix_posneg = get_dependence_matrix(data, dependence_method='optimal_transport',
                                            target_dependence='positive_negative')

Research Notebooks
##################

The following research notebook can be used to better understand the optimal transport distance measure described above.

* `Optimal Transport`_

.. _`Optimal Transport`: https://github.com/hudson-and-thames/research/blob/master/Codependence/Optimal%Transport/optimal_transport.ipynb