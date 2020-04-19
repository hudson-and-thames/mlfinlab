.. _implementations-clustering:


==========
Clustering
==========


Abstract
########

Clustering is a task of grouping a set of data points into several groups where each group contains more similar data points compared to those in other groups. One of the most popular clusterings is the K-means algorithm. K-means algorithm has an advantage that it guarantees its convergence in a finite number of steps; thus, it gives an assurance of a stable result. 

However, the K-means algorithm requires the user to set the number of clusters in advance. Furthermore, the initialization of the clusters is random. As consequent, the solution is often not optimal in some sense, and the effectiveness of the algorithm can be random. 

To address this problem, Prof. Marcos M. Lopez de Prado designed a k-means based algorithm called the Optimal Number of Clusters (ONC). The algorithm accommodates quality assessment features that ensure the optimality of the algorithm's clustering.

Optimal Number of Clusters (ONC)
################################

Optimal Number of Clusters algorithm detects the optimal number of K-Means clusters using feature correlation matrix and silhouette scores. This implementation is based on 'Detection of False Investment Strategies using Unsupervised Learning Methods' https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

The same subject and information can also be found in 'Machine Learning for Asset Managers', Marcos M. Lopez de Prado, Cornell University, New York, April 2020, Chapter 4, Page 52-64.

This algorithm results in a tuple that contains:

        * An optimized correlation matrix
        * Optimized clusters
        * The Silhouette Scores of every node in the clusters


Formulas
********

The ONC algorithm accommodates three distinct formulas that set the algorithm apart from the traditional K-means clustering. These formulas are:

* Silhouette Scores:

        :math:`S_i=\frac{b_i-a_i}{max\{a_i,b_i\}}`

        :math:`a_i` = the average distance between *i* and all other components in the same cluster

        :math:`b_i` = the average distance between *i* and all the components in the nearest cluster where *i* is not included

* The measure of clustering quality or *q*:

        :math:`\textit q= \frac{E[\{S_i\}]}{\sqrt{V[\{S_i\}]}}`

        :math:`E[\{S_i\}]` = the mean of the Silhouette coefficients 

        :math:`V[\{S_i\}]` = the variance of the Silhouette coefficients

* The distance matrix formula:

        :math:`D_{i,j}= \sqrt{\frac{1}{2}(1-\rho_{ij})}`

        :math:`\rho_{ij}` = the correlation between entities *i* and *j*


        
The ONC Mechanism
*****************

We divide the ONC algorithm into two main parts, the Base Clustering, and the Top- Level Clustering. The mechanism of the parts is available below.

**The Base Clustering**

.. image:: clustering_images/ONC_diagram_base.png
   :scale: 100 %
   :align: center 

Figure 4.1.  Marcos M. Lopez de Prado. *Structure of ONC's base clustering stage*. 2020. Machine Learning for Asset Managers. Marcos M. Lopez de Prado. Cornell University, New York. April 2020. Page 57. Digital Book.

First, we feed an observation matrix into the algorithm. The Base Clustering evaluates the observation matrix before performing double for..loop on the matrix. The first loop clusters the matrix for different k values of clusters from 2 to N via k-means for one given initialization and assesses the quality q for each clustering. The second loop redoes the first loop multiple times until it satisfies the condition of different initializations. Finally, the algorithm chooses the clustering with the highest quality *q* over these two loops.  


**The Top-Level Clustering**

.. image:: clustering_images/ONC_diagram.png
   :scale: 100 %
   :align: center 

Figure 4.2. Marcos M. Lopez de Prado and Lewis (2018). *Structure of ONC's higher-level stage*. 2020. Machine Learning for Asset Managers. Marcos M. Lopez de Prado. Cornell University, New York. April 2020. Page 60. Digital Book. 

The second algorithm, the Top- Level of Clustering, evaluates the quality of each cluster of the optimum clustering from the first algorithm. Then, the algorithm takes the average quality of the clusters and finds the set of clusters with below-average quality. We mark the number of sub-par quality clusters :math:`K_1` where :math:`K_1` < :math:`K` (number of clusters). The next step is to rerun the clustering of the items in the :math:`K_1` clusters if :math:`K_1 >=2` or to return the clustering given by the base algorithm if :math:`K_1 <=1`. 

We rerun the Base Clustering algorithm on the matrix that is composed of the elements of :math:`K_1` clusters. This process may return a new clustering for the elements in :math:`K_1`. Then, we check the efficacy of the new clustering by comparing the average cluster quality before and after reclustering the elements in :math:`K_1`. If the average cluster quality does not improve, then we return the clustering created by the initial Base Clustering algorithm. However, if we see an improvement in the average cluster quality, then we return the accepted clustering from the initial Base Clustering concatenated with the newly redone clustering. 


Implementation
##############

.. py:currentmodule:: mlfinlab.clustering.onc


.. automodule:: mlfinlab.clustering.onc
   :members: get_onc_clusters
           
  
Example
#######

Assuming that we want to find the optimum clustering of a matrix by using the ONC algorithm. Let's create the necessary correlation matrix.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mlfinlab.clustering.onc import get_onc_clusters # import the ONC function
    
    data = [[1,0.5,-0.2,0.7,0], [0.5,1,-1,0,-0.5], [-0.2,-1,1,0.1,0.8],\
            [0.7,0,0.1,1,-0.5], [0,-0.5,0.8,-0.5,1]] 
    df = pd.DataFrame(data) 
    df


Apply ONC algorithm on the matrix.

.. code-block:: python
    
    get_onc_clusters(df,10)

The algorithm results in a tuple that contains the optimized correlation matrix, the optimized clusters, and the Silhouette Scores of every node in the clusters. The result can be seen below.

::

        (     2    4    0    1    3
         2  1.0  0.8 -0.2 -1.0  0.1
         4  0.8  1.0  0.0 -0.5 -0.5
         0 -0.2  0.0  1.0  0.5  0.7
         1 -1.0 -0.5  0.5  1.0  0.0
         3  0.1 -0.5  0.7  0.0  1.0, {0: [2, 4], 1: [0, 1, 3]}, 0    0.435986
         1    0.366015
         2    0.620819
         3    0.336439
         4    0.617621
         dtype: float64)


