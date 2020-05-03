.. _implementations-feature_clusters:

================
Feature Clusters
================

This module implements the clustering of features to generate a feature subset described in the book Machine Learning for
Asset Manager (snippet 6.5.2.1 page-85). This subsets can be further utilised for getting Clustered Feature Importance
using the clustered_subsets argument in the Mean Decreased Impurity (MDI) and Mean Decreased Accuracy (MDA) algorithm.

The algorithm projects the observed features into a metric space by applying the dependence metric function, either correlation
based or information theory based (see the codependence section). Information-theoretic metrics have the advantage of
recognizing redundant features that are the result of nonlinear combinations of informative features.

Next, we need to determine the optimal number of clusters. The user can either specify the number cluster to use, this will apply a
hierarchical clustering on the defined distance matrix of the dependence matrix for a given linkage method for clustering,
or the user can use the ONC algorithm which uses K-Means clustering, to automate these task.

The caveat of this process is that some silhouette scores may be low due to one feature being a combination of multiple features across clusters.
This is a problem, because ONC cannot assign one feature to multiple clusters. Hence, the following transformation may help
reduce the multicollinearity of the system:

For each cluster :math:`k = 1 . . . K`, replace the features included in that cluster with residual features, so that it
do not contain any information outside cluster :math:`k`. That is let :math:`D_{k}` be the subset of index
features :math:`D = {1,...,F}` included in cluster :math:`k`, where:

.. math::
    D_{k}\subset{D}\ , ||D_{k}|| > 0 \ , \forall{k}\ ; \ D_{k} \bigcap D_{l} = \Phi\ , \forall k \ne l\ ; \bigcup \limits _{k=1} ^{k} D_{k} = D

Then, for a given feature :math:`X_{i}` where :math:`i \in D_{k}`, we compute the residual feature :math:`\hat \varepsilon _{i}`
by fitting the following equation for regression:

.. math::
      X_{n,j} = \alpha _{i} + \sum \limits _{j \in \bigcup _{l<k}} \ D_{l} \beta _{i,j} X_{n,j} + \varepsilon _{n,i}

Where :math:`n = 1,\dots,N` is the index of observations per feature. Note if the degrees of freedom in the above regression
are too low, one option is to use as regressors linear combinations of the features within each cluster by following a
minimum variance weighting scheme so that only :math:`K-1` betas need to be estimated. This transformation is not necessary
if the silhouette scores clearly indicate that features belong to their respective clusters.

Implementation
**************

This module creates clustered subsets of features described in the presentation slides: `Clustered Feature Importance <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595>`_
by Marcos Lopez de Prado.

.. py:currentmodule:: mlfinlab.clustering.feature_clusters

.. autofunction:: get_feature_clusters

Example
*******

An example showing how to generate feature subsets or clusters for a give feature DataFrame.
The example will generate 4 clusters by Hierarchical Clustering for given specification.

.. code-block::

    import pandas as pd
    from mlfinlab.clustering.feature_clusters import get_feature_clusters

    # Read the a csv file containing only features
    X = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

    feat_subs = get_feature_clusters(X, dependence_metric='information_variation',
                                     distance_metric='angular', linkage_method='singular',
                                     n_clusters=4)
Research Notebook
*****************
The for better understanding of its implementations see the notebook on Clustered Feature Importance.

.. _Clustered Feature Importance: https://github.com/hudson-and-thames/research/blob/master/Chapter8_FeatureImportance/Cluster_Feature_Importance.ipynb
