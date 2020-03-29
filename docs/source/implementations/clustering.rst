.. _implementations-clustering:

================================
Optimal Number of Clusters (ONC)
================================

Optimal Number of Clusters algorithm detects optimal number of K-Means clusters using feature correlation matrix and silhouette scores.
This implementation is based on 'Detection of False Investment Strategies using Unsupervised Learning Methods'
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017


Implementation
==============

.. py:currentmodule:: mlfinlab.clustering.onc
.. automodule:: mlfinlab.clustering.onc
   :members: get_onc_clusters


The ONC algorithm workflow
==========================


1. The module will initially transform the correlated numbers of investment strategies into correlated distance numbers

2. The process will take two steps of creating distance numbers:
    
   calculating the proper distance matrix:
.. image:: https://raw.githubusercontent.com/leren123/mlfinlab/onc_docs/docs/source/implementations/labeling_images/proper_distance_matrix_formula.PNG
   :scale: 100 %
   :align: center 

where  rho i,j is the correlation returns between strategies i and j

calculating the Euclidean distance matrix of the proper distance matrix:

.. image:: https://raw.githubusercontent.com/leren123/mlfinlab/onc_docs/docs/source/implementations/labeling_images/Euclidean_distance_formula.PNG
   :scale: 100 %
   :align: center 
 
where D is the the direct proper distance number of rho

3. The algorithm then will do the clustering of our distance matrix through K-means algorithm

4. Silhouette scores will be calculated from the distance numbers
   
the formula to find the Silhouette score is as follows:
	
.. image:: https://raw.githubusercontent.com/leren123/mlfinlab/onc_docs/docs/source/implementations/labeling_images/silhouette_score_formula.PNG
   :scale: 100 %
   :align: center 
 

where ai is the average distance between i and all other nodes in the same cluster, 
and bi is the smallest average distance between i and all the nodes in any other cluster

5. Then the measure of quality q for each clustering is calculated as follows:

.. image:: https://raw.githubusercontent.com/leren123/mlfinlab/onc_docs/docs/source/implementations/labeling_images/quality_distance_formula.PNG
   :scale: 100 %
   :align: center 
 

6. Step 3 is done in a for.. loop manner in which the first loop clusters an initialization that is evaluated by the quality of each clustering

7. The second loop repeats the first loop multiple times until it results in different initializations

8. Then the module chooses the clustering with the highest quality measure, the process is known as the base clustering

9. Further clustering is done in the next step in order for the module to evaluate the quality of each cluster k=1,...,K 
   given the clustering and silhouette scores obtained from the base clustering algorithm

10. We then take the average quality value and find the set of clusters with below average quality

11. The number of clusters in the set,K1 < K,  then are tested by the conditions of:
    - If the number of clusters to rerun is K1 <= 2, then we return the clustering that is given by the base algorithm
    - If K1 > 2 then we rerun the clustering of the items in those K1 clusters, while the rest are considered as acceptably      clustered

12. The process will possibly return a new optimal clustering for the nodes

13. The system will check whether the average quality of the clusters improve

14. The module will return the latest clustering that is concatenated to base clustering

15. Otherwise we return the clustering formed by the base algorithm


Example
=======

Optimal Number of Clusters (ONC)

Optimal Number of Clusters algorithm detects optimal number of K-Means clusters using feature correlation matrix and silhouette scores.
This implementation is based on 'Detection of False Investment Strategies using Unsupervised Learning Methods' https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

The result of this algorithm is a tupple that contains:

1. Correlation Matrix
2. Optimized Clusters
3. Silhouette Scores

Correlation Matrix show the matrix that are sorted by their relevance. Optimized Clustres show the optimal number of clustres and each of the culsters' contents.

::

    import numpy as np
    import pandas as pd
    from mlfinlab.clustering.onc import get_onc_clusters # import the ONC function	
::
    
    data = [[1,0.5,-0.2,0.7,0], [0.5,1,-1,0,-0.5], [-0.2,-1,1,0.1,0.8], [0.7,0,0.1,1,-0.5], [0,-0.5,0.8,-0.5,1]] 
    df1 = pd.DataFrame(data) 
    df1
    
Assuming we have a correlation matrix data as in the above

::
    
    get_onc_clusters(df,10)
