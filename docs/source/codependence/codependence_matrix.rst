.. _codependence-codependence_matrix:

.. note::
   The following implementations and documentation, closely follows the lecture notes notes from Cornell University, by Marcos Lopez de Prado:
   `Codependence (Presentation Slides) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_.

===================
Codependence Matrix
===================

This module consists two functions that generate the following:

1. **Dependence Matrix** to compute dependence of a given matrix using various codependence methods like Mutual Information,
   Variation of Information, Distance Correlation, Spearman's Rho, GPR distance, and GNPR distance.
2. **Distance Matrix** can used to compute distance of a given matrix using various metrics like angular, squared
   angular and absolute angular.

.. note::

   MlFinLab makes use of these functions in the clustered feature importance.

   The Spearman's Rho, GPR distance, and GNPR distance are described in the **Codependence by Marti** section of the docs.

Implementation
**************

.. py:currentmodule:: mlfinlab.codependence.codependence_matrix
.. autofunction:: get_dependence_matrix
.. autofunction:: get_distance_matrix


Example
*******

.. code-block::

   import pandas as pd
   from mlfinlab.codependence.codependence_matrix import (get_dependence_matrix,
                                                          get_distance_matrix)

   X = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

   dep_matrix = get_dependence_matrix(X, dependence_method='distance_correlation')
   dist_matrix = get_distance_matrix(dep_matrix, distance_metric='angular')
