.. _implementations-clustering:

================================
Optimal Number of Clusters (ONC)
================================

Optimal Number of Clusters algorithm detects optimal number of K-Means clusters using feature correlation matrix and silhouette scores.
This implementation is based on 'Detection of False Investment Strategies using Unsupervised Learning Methods' https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

.. py:currentmodule:: mlfinlab.clustering.onc
.. automodule:: mlfinlab.clustering.onc
   :members: get_onc_clusters


**Abstract**

The module will cluster investment strategies based on their correlation among each other. 
The algorithm will search for the optimal number of clusters within the correlation matrix. 


**Steps for implementation and explanation**
- Import the necessary packages
- Feed in a dataframe that contains correlation numbers among its nodes and the number of clustering repeat into the module parameters
- The algorithm will create 
- 
