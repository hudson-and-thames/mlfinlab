.. _codependence-optimal_trnsport:

.. note::
   The following documentation closely follows the work by Marti et al.:
   `Exploring and measuring non-linear correlations: Copulas, Lightspeed Transportation and Clustering <https://arxiv.org/pdf/1610.09659.pdf>`_.

   Initial implementation was taken from the blog post by Gautier Marti:
   `Measuring non-linear dependence with Optimal Transport <https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html>`_.

=================
Optimal Transport
=================

...

Optimal Transport distance
==========================

...

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