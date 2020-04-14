.. _implementations-clustering:

================================
Optimal Number of Clusters (ONC)
================================

The ONC algorithm detects the optimal number of K-Means clusters using a correlation matrix as input.

This implementation is based on the paper: `López de Prado, M. and Lewis, M.J., 2019. Detection of false investment
strategies using unsupervised learning methods. Quantitative Finance, 19(9), pp.1555-1565. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017>`_

Implementation
##############

.. py:currentmodule:: mlfinlab.clustering.onc
.. autofunction:: get_onc_clusters
