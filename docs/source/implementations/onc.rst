.. _implementations-clustering:

================================
Optimal Number of Clusters (ONC)
================================

The ONC algorithm detects the optimal number of K-Means clusters using a correlation matrix as input.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Detection of false investment strategies using unsupervised learning methods** *by* Marcos Lopez de Prado *and* Lewis, M.J. `available here <https://papers.ssrn.com/sol3/abstract_id=3167017>`__. *Describes the ONC algorithm in detail. The code in this module is based on the code written by the researchers.*

The ONC algorithm structure is described in the work **Detection of false investment strategies using unsupervised learning methods**
using the following diagrams:

.. image:: clustering_images/onc_base_clustering.png
   :scale: 100 %
   :align: center

In the base clustering stage first the distances between the elements are calculated, then the algorithm iterates through
a set of possible number of clusters :math:`N` times. For each iteration, a clustering result is evaluated using t-statistic of
the silhouette scores.

The clustering result with the best silhouette score is picked, the correlation matrix is reordered so that clustered elements
are positioned close to each other.

.. image:: clustering_images/onc_higher_level.png
   :scale: 100 %
   :align: center

On a higher level, the average t-score of the clusters from the base clustering stage is calculated. If more than three
clusters have a t-score below average, these clusters go through the base clustering stage again.

Then, based on the t-statistic it is checked whether the new clustering has improved the original one. The output of the
algorithm is the best clustering result, reordered correlation matrix, and silhouette scores.

Implementation
##############

.. py:currentmodule:: mlfinlab.clustering.onc
.. autofunction:: get_onc_clusters

Example
#######

An example showing how the NCO algorithm is used can be seen below:

.. code-block::

    import pandas as pd
    from mlfinlab.clustering import ONC

    # Import dataframe of returns for assets
    asset_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Calculate correlation matrix of returns
    assets_corr = asset_returns.corr()

    # Class that contains needed function
    onc = ONC()

    # Output of the ONC algorithm with 10 simulations for each number of clusters tested
    assets_corr_onc, clusters, silhscores = onc.get_onc_clusters(assets_corr, repeat=10)
