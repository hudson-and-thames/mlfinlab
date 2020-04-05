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


1. The module initially transforms the correlated numbers of investment strategies into correlated distance numbers

2. The process takes two steps of creating distance numbers
    
3. Calculating the proper distance matrix:
   
:math:`D_{i,j}= \sqrt{\frac{1}{2}(1-\rho_{ij})}` 



4. Calculating the Euclidean distance matrix of the proper distance matrix:

:math:`\tilde{D}_{i,j}=\sqrt{\sum\limits_{k}(D_{ik}-D_{jk})^2}` 

       
5. The algorithm does the clustering with K-means algorithm that is modified by involving Silhouette scores and the measure of quality for each clustering

6. Silhouette scores are calculated from the distance numbers
   
7. the formula to find the Silhouette scores is as follows:
		
 
:math:`S_i=\frac{b_i-a_i}{max\{a_i,b_i\}}`



8. Then the measure of quality q for each clustering is calculated as follows:

 
:math:`\textit q= \frac{E[\{S_i\}]}{\sqrt{V[\{S_i\}]}}` 


9. Second modification on K-mean's that involves double for.. loop is performed

10. In the first loop, we cluster each different k=2,...,N-1 via K-means for one given initialization and evaluate the quality q of each clustering

11. The second loop repeats the first loop multiple times that results in different initializations

12. Then the module chooses the clustering with the highest quality measure

13. The third modification to K-means deals with clusters of inconsistent quality, each cluster's quality or qk (k=1,...,K) is evaluated 

14. We then take the average quality value and find the set of clusters with below average quality

15. The number of clusters in the set,K1 < K,  then are tested by the conditions of:
    - If the number of clusters to rerun is K1 <= 2, then we return the clustering that is given by the base algorithm
    - If K1 > 2 then we rerun the clustering of the items in those K1 clusters, while the rest are considered as acceptably clustered

16. We rerun the K1 clusters in a recursive manner, rerunning the clustering on the matrix, restricted to the nodes in the K1 clusters

17. The process possibly returns a new optimal clustering for the nodes

18. To check its efficacy, we compare the average quality of the clusters to redo given the previous clustering to the average quality of the clusters given the new clustering

19. If the average quality improves for these clusters, we return the accepted clustering from the base clustering concatenated with the new clustering for the nodes redone

20. Otherwise we return the clustering formed by the base algorithm


Example
=======

Optimal Number of Clusters (ONC)

ONC is an algorithm that will partition N series of correlations into an optimal number of clusters.

The result of this algorithm is a tupple that contains:

1. Correlation Matrix
2. Optimized Clusters
3. Silhouette Scores

Correlation Matrix show the matrix that are sorted by their relevance. Optimized Clusters show the optimal number of clusters and each of the clusters' contents.

::

    import numpy as np
    import pandas as pd
    from mlfinlab.clustering.onc import get_onc_clusters # import the ONC function	
::
    
    data = [[1,0.5,-0.2,0.7,0], [0.5,1,-1,0,-0.5], [-0.2,-1,1,0.1,0.8], [0.7,0,0.1,1,-0.5], [0,-0.5,0.8,-0.5,1]] 
    df = pd.DataFrame(data) 
    df
    
Assuming that we have a correlation matrix data as in the above

::
    
    get_onc_clusters(df,10)



