.. _portfolio_optimisation-theory_implied_correlation:


================================
Theory-Implied Correlation (TIC)
================================
This TIC class includes an algorithm to calculate the Theory-Implied Correlation and a method to calculate the correlation matrix distance proposed by Herdin and Bonek. This distance may be used to measure to which extent the TIC matrix has blended theory-implied views (tree structure of the elements) with empirical evidence (correlation matrix).

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Estimation of Theory-Implied Correlation Matrices** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/abstract_id=3484152>`__. *Describes the TIC algorithm*
   - **AMIMO Correlation Matrix based Metric for Characterizing Non-Stationarity** *by* Markus Herdin *and* Ernst Bonek `available here <https://publik.tuwien.ac.at/files/pub-et_8791.pdf>`__. *Describes the Correlation Matrix Distance metric*

About the Algorithm
###################

The TIC algorithm is aiming to estimate a forward-looking correlation matrix based on economic theory. The method is using a theoretical classification of assets (hierarchical structure) and fits the empirical correlation matrix to the theoretical structure.

From the work **Estimation of Theory-Implied Correlation Matrices**:
"A problem of empirical correlation matrices is that they are purely observation driven, and do not impose a structural view of the investment universe, supported by economic theory."

Using the TIC approach allows us to include forward-looking views to the world, instead of only backward-looking views from the empirical correlations matrix.

The economic theory in the algorithm is represented in terms of a tree graph. "The tree can include any number of levels needed, each branch should have one or more leaves, some branches may include more levels than others".

An example of how the theoretical structure can be used is the `MSCI's Global Industry Classification Standard (GICS) <https://www.msci.com/gics>`__ for investments. Using this structure, each stock can be classified using four levels of depth.

To use a tree as the input to the algorithm, it should have the "bottom-up order of the columns, where the leftmost column is corresponding to terminal leaves and the rightmost columns corresponding to the tree's root". An  example of a tree graph according to the GICS:

+---------------+----------------+-----------+-----------------+---------+
| Ticker        | Sub-Industry   | Industry  | Industry Group  | Sector  |
+===============+================+===========+=================+=========+
| A UN Equity   | 35203010       | 352030    | 3520            | 35      |
+---------------+----------------+-----------+-----------------+---------+
| AAL UW Equity | 20302010       | 203020    | 2030            | 20      |
+---------------+----------------+-----------+-----------------+---------+
| AAP UN Equity | 25504050       | 255040    | 2550            | 25      |
+---------------+----------------+-----------+-----------------+---------+
| AAPL UW Equity| 45202030       | 452020    | 4520            | 45      |
+---------------+----------------+-----------+-----------------+---------+

The empirical correlation matrix used in the TIC algorithm is estimated on historical observations. "It should be symmetric and have a main diagonal of 1s. But it doesn't need to be invertible, positive-definite or non-singular".

Steps of the Algorithm
######################

The TIC algorithm consists of three steps:

1. In the first step, the theoretical tree graph structure of the assets is fit on the evidence presented by the empirical correlation matrix.

 - If there is no top level of the tree (tree root), this level is added so that all variables are included in one general cluster.

 - The empirical correlation matrix is transformed into a matrix of distances using the above formula:

 .. math::
    d_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{i,j})}

 - For each level of the tree, the elements are grouped by elements from the higher level. The algorithm iterates from the lowest to the highest level of the tree.

 - A linkage object is created for these grouped elements based on their distance matrix using the `SciPy linkage function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`__. Each link in the linkage object is an array representing a cluster of two elements and has the following data as elements:

  - ID of the first element in a cluster

  - ID of the second element in a cluster

  - Distance between the elements

  - Number of atoms (simple elements from the portfolio and not clusters) inside

 - A linkage object is transformed to reflect the previously created clusters.

 - A transformed local linkage object is added to the global linkage object

 - The distance matrix is adjusted to the newly created clusters - elements that are now in the new clusters are replaced by the clusters in the distance matrix. The distance from the new clusters to the rest of the elements in the distance matrix is calculated as a weighted average of distances of two elements in a cluster to the other elements. The weight is the number of atoms in an element. So, the formula is:

 .. math::
    DistanceCluster = \frac{Distance_1 * NumAtoms_1 + Distance_2 * NumAtoms_2}{NumAtoms_1 + NumAtoms_2}

 - The linkage object, representing a dendrogram of all elements in a portfolio is the result of the first step of the algorithm. It sequentially clusters two elements together, while measuring how closely together the two elements are, until all elements are subsumed within the same cluster.

2. In the second step, a correlation matrix is derived from the linkage object.

 - One by one, the clusters (each represented by a link in the linkage object) are decomposed to lists of atoms contained in each of the two elements of the cluster.

 - The elements on the main diagonal of the resulting correlation matrix are set to 1s. The off-diagonal correlations between the variables are computed as:

 .. math::
    \rho_{i,j} = 1 - 2 * d_{i,j}^{2}

3. In the third step, the correlation matrix is de-noised.

 - The eigenvalues and eigenvectors of the correlation matrix are calculated.

 - Marcenko-Pastur distribution is fit to the eigenvalues of the correlation matrix and the maximum theoretical eigenvalue is calculated.

 - This maximum theoretical eigenvalue is set as a threshold and all the eigenvalues above the threshold are shrinked.

 - The de-noised correlation matrix is calculated back from the eigenvectors and the new eigenvalues.

.. tip::

    * The algorithm for de-noising the correlation and the covariance matrix is implemented in the RiskEstimators class of the mlfinlab package. It is described in more detail `here <https://github.com/hudson-and-thames/research/blob/master/RiskEstimators/RiskEstimators.ipynb>`__.

    * This algorithm is described in more detail in the paper **Estimation of Theory-Implied Correlation Matrices** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/abstract_id=3484152>`__.

Correlation Matrix Distance
###########################

The similarity of the Empirical correlation matrix and the Theory-implied correlation matrix can be measured using the correlation matrix distance introduced by Herdin and Bonek.

The distance is calculated as:

.. math::
   d[\sum_{1},\sum_{2}] = 1 - \frac{tr(\sum_{1}\sum_{2})}{||\sum_{1}||_f||\sum_{2}||_f}

Where :math:`\sum_{1},\sum_{2}` are the two correlation matrices and the :math:`||.||_f` is the Frobenius norm.

From the work **Estimation of Theory-Implied Correlation Matrices**:
"The distance :math:`\sum_{1},\sum_{2}` measures the orthogonality between the considered correlation matrices. It becomes zero if the correlation matrices are equal up to a scaling factor, and one if they differ to a maximum extent".

.. tip::

    The Correlation Matrix Distance metric is described in more detail in the work **AMIMO Correlation Matrix based Metric for Characterizing Non-Stationarity** *by* Markus Herdin *and* Ernst Bonek `available here <https://publik.tuwien.ac.at/files/pub-et_8791.pdf>`__.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.tic

    .. autoclass:: TIC
        :members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.tic import TIC

    # Reading data
    tree_classification = pd.read_csv('TREE_FILE_PATH')
    stock_returns = pd.read_csv('DATA_FILE_PATH', parse_dates=True, index_col='Date')

    # Calculating the empirical correlation matrix
    corr_matrix = stock_returns.corr()

    # Calculating the relation of sample length T to the number of variables N
    # It's used for de-noising the TIC matrix
    tn_relation = stock_returns.shape[0] / stock_returns.shape[1]

    # The class that contains the TIC algorithm
    tic = TIC()

    # Calculating the Theory-Implied Correlation matrix
    tic_matrix = tic.tic_correlation(tree_classification, corr_matrix, tn_relation, kde_bwidth=0.01)

    # Calculating the distance between the empirical and the theory-implied correlation matrices
    matrix_distance = tic.corr_dist(corr_matrix, tic_matrix)

Research Notebooks
##################

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Theory-Implied Correlation Notebook`_

.. _Theory-Implied Correlation Notebook: https://github.com/hudson-and-thames/research/blob/master/TIC/TIC.ipynb

