.. _implementations-feature_clusters:

================
Feature Clusters
================

This module implements the clustering of features to generate a feature subset. This subsets can be further utilised for getting Clustered Feature Importance using the clustered_subsets argument in the Mean Mean Decrease Accuracy (MDA) algorithm.


Implementation
**************

.. py:currentmodule:: mlfinlab.clustering.feature_clusters
.. automodule:: mlfinlab.clustering.feature_clusters
   :members: get_feature_clusters

Example
*******

An example showing how to generate feature subsets or clusters for a give feature DataFrame.

.. code-block::

    import pandas as pd
    from mlfinlab.clustering.feature_clusters import get_feature_clusters

    #Read the a csv file containing only features
    X = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

    feat_subs = get_feature_clusters(X, dependence_metric='information_variation', distance_metric='angular',
                                          linkage_method='singular', n_clusters=4)
