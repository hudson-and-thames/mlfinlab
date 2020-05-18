.. _implementations-codependence_marti:


=====================
Codependence by Marti
=====================



Correlation Metrics
###################



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
