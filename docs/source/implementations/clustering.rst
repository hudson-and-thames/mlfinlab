.. _implementations-clustering:
==========
Clustering
==========

Optimal Number of Clusters (ONC)
================================

Optimal Number of Clusters algorithm detects optimal number of K-Means clusters using feature correlation matrix and silhouette scores.
This implementation is based on 'Detection of False Investment Strategies using Unsupervised Learning Methods'
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017


.. py:currentmodule:: mlfinlab.clustering.onc
.. automodule:: mlfinlab.clustering.onc
   :members: get_onc_clusters


Feature Clusters
================

.. py:currentmodule:: mlfinlab.clustering.feature_clusters
.. automodule:: mlfinlab.clustering.feature_clusters
   :members: get_feature_clusters

::

    import pandas as pd
    from mlfinlab.clustering.feature_clusters import get_feature_clusters

    X = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

    feat_subs = get_feature_clusters(X, dependence_metric='information_variation', distance_metric='angular',
                                          linkage_method='singular', n_clusters=4)
::
