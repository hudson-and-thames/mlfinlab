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


Initially, the algorithm was developed to clusterize investment strategies backtests, but it can be used to clusterize features also.


1. The module initially transforms the correlated numbers of investment strategies into correlated distance numbers

2. The process takes two steps of creating distance numbers

3. Calculating the distance matrix:

:math:`D_{i,j}= \sqrt{\frac{1}{2}(1-\rho_{ij})}`



4. Calculating the Euclidean distance matrix of the distance matrix:


:math:`\tilde{D}_{i,j}=\sqrt{\sum\limits_{k}(D_{ik}-D_{jk})^2}`


5. The algorithm does the clustering with K-means algorithm that is modified by involving Silhouette scores and the measure of quality for each clustering

6. Silhouette scores are calculated from the distance numbers

7. the formula to find the Silhouette scores is as follows:


:math:`S_i=\frac{b_i-a_i}{max\{a_i,b_i\}}`



8. Then the measure of quality q for each clustering is calculated as follows:


:math:`\textit q= \frac{E[\{S_i\}]}{\sqrt{V[\{S_i\}]}}`


9. Second modification on K-mean's initialization problem that involves double for.. loop is performed

10. Initially,at the base level,  we are given an evaluated N x N correlation matrix that from which the distance matrix and euclidean distance matrix were evaluated

11. In the first loop, we cluster the matrix with each different k=2,...,N-1 via K-means for one given initialization and evaluate the quality q of each clustering

12. The second loop repeats the first loop several times that obtains different initializations

13. Then the module chooses the clustering with the highest quality measure

14. The next modification to K-means deals with clusters that have inconsistent quality, each cluster's quality or qk (k=1,...,K) is evaluated with the given clustering and silhouette scores that are taken from the base clustering algorithm

15. We then take the average quality value and we find the set of clusters with below average quality

16. We then test The number of clusters in the set,K1 < K, under the conditions of:

    - If the number of clusters to rerun is K1 <= 2, then we return the clustering that is given by the base algorithm

    - If K1 > 2 then we rerun the clustering of the K1 clusters' entities, while the rest are accepted clusters

17. We rerun the K1 clusters with a recursive way, rerunning the clustering on the matrix, limited to the nodes in the K1 clusters

18. The process possibly returns a new optimal clustering for the nodes

19. Then we check its efficacy, by comparing the average quality of the clusters to redo given the previous clustering to the average quality of the clusters given the new clustering

20. If the clusters have improved average quality then we return the accepted clustering from the base clustering with the concatenated new clustering for the nodes redone

21. If not so, we return the clustering formed by the base algorithm



The ONC algorithm diagram
==========================

Structure of ONC's base clustering stage.

.. image:: clustering_images/ONC_diagram_base.png
   :scale: 100 %
   :align: center 

Figure 4.1.  Marcos M. Lopez de Prado. *Structure of ONC's base clustering stage*. 2020. MACHINE LEARNING FOR ASSET MANAGERS. Marcos M. Lopez de Prado. Cornell University, New York. April 2020. Page 57. Digital Book.   

Structure of ONC's higher-level stage.

.. image:: clustering_images/ONC_diagram.png
   :scale: 100 %
   :align: center 

Figure 4.2.  Lopez de Prado and Lewis. *Structure of ONC's higher-level stage*. 2020. MACHINE LEARNING FOR ASSET MANAGERS. Marcos M. Lopez de Prado. Cornell University, New York. April 2020. Page 60. Digital Book.   
 

Example
=======

Optimal Number of Clusters (ONC)

The result of this algorithm is a tupple that contains:

1. Correlation Matrix
2. Optimized Clusters
3. Silhouette Scores

Correlation Matrix shows the matrix that is sorted under the optimization clustering. Optimized Clusters show the optimal number of clusters and each of the clusters' contents. Silhouette Scores show Silhouette quality of each node in the clusters.

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



Tips
=======

.. tip::

   Did you know that you can also apply ONC algorithm on any kind of matrix apart from correlation matrix?

.. tip::
   
   ONC will work directly with any matrix data, so, feel free to put in your matrix directly to the function.

.. tip::
   
   You do not have to assume or decide any number of initial clusters with this algorithm, so sit back, relax, and enjoy the clustering!




    
