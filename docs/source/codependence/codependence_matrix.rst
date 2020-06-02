.. _codependence-codependence_matrix:

===================
Codependence Matrix
===================

The functions in this part of the module are used to generate dependence and distance matrices using the codependency and
distance metrics described previously.

1. **Dependence Matrix** function is used to compute dependence of a given dataframe of elements using various codependence
   methods like Mutual Information, Variation of Information, Distance Correlation, Spearman's Rho, GPR distance,
   and GNPR distance.

2. **Distance Matrix** function can be used to compute distance of a given codependency matrix using distances like
   angular, squared angular and absolute angular.

.. note::

   MlFinLab makes use of these functions in the clustered feature importance and portfolio optimization modules.

Implementation
==============

.. py:currentmodule:: mlfinlab.codependence.codependence_matrix
.. autofunction:: get_dependence_matrix
.. autofunction:: get_distance_matrix


Example
=======

.. code-block::

   import pandas as pd
   from mlfinlab.codependence.codependence_matrix import (get_dependence_matrix,
                                                          get_distance_matrix)

    # Import dataframe of returns for assets in a portfolio
    asset_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Calculate distance correlation matrix
    distance_corr = get_dependence_matrix(asset_returns, dependence_method='distance_correlation')

    # Calculate Pearson correlation matrix
    pearson_corr = asset_returns.corr()

    # Calculate absolute angular distance from a Pearson correlation matrix
    abs_angular_dist = absolute_angular_distance(pearson_corr)
