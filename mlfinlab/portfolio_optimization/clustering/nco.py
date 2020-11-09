# pylint: disable=missing-module-docstring
from math import ceil
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from mlfinlab.portfolio_optimization.estimators import RiskEstimators


class NestedClusteredOptimisation:
    """
    This class implements the Nested Clustered Optimization (NCO) algorithm, the Convex Optimization Solution (CVO),
    the  Monte Carlo Optimization Selection (MCOS) algorithm and sample data generating function. It is reproduced with
    modification from the following paper: `Marcos Lopez de Prado “A Robust Estimator of the Efficient Frontier”,
    (2019). <https://papers.ssrn.com/abstract_id=3469961>`_.
    """

    def __init__(self):
        """
        Initialize
        """


        pass

    def allocate_cvo(self, cov, mu_vec=None):
        """
        Estimates the Convex Optimization Solution (CVO).

        Uses the covariance matrix and the mu - optimal solution.
        If mu is the vector of expected values from variables, the result will be
        a vector of weights with maximum Sharpe ratio.
        If mu is a vector of ones, the result will be a vector of weights with
        minimum variance.

        :param cov: (np.array) Covariance matrix of the variables.
        :param mu_vec: (np.array) Expected value of draws from the variables for maximum Sharpe ratio.
                              None if outputting the minimum variance portfolio.
        :return: (np.array) Weights for optimal allocation.
        """

        pass

    def allocate_nco(self, cov, mu_vec=None, max_num_clusters=None, n_init=10):
        """
        Estimates the optimal allocation using the nested clustered optimization (NCO) algorithm.

        First, it clusters the covariance matrix into subsets of highly correlated variables.
        Second, it computes the optimal allocation for each of the clusters separately.
        This allows collapsing of the original covariance matrix into a reduced covariance matrix,
        where each cluster is represented by a single variable.
        Third, we compute the optimal allocations across the reduced covariance matrix.
        Fourth, the final allocations are the dot-product of the intra-cluster (step 2) allocations and
        the inter-cluster (step 3) allocations.

        For the Convex Optimization Solution (CVO), a mu - optimal solution parameter is needed.
        If mu is the vector of expected values from variables, the result will be
        a vector of weights with maximum Sharpe ratio.
        If mu is a vector of ones (pass None value), the result will be a vector of weights with
        minimum variance.

        :param cov: (np.array) Covariance matrix of the variables.
        :param mu_vec: (np.array) Expected value of draws from the variables for maximum Sharpe ratio.
                              None if outputting the minimum variance portfolio.
        :param max_тum_сlusters: (int) Allowed maximum number of clusters. If None then taken as num_elements/2.
        :param n_init: (float) Number of time the k-means algorithm will run with different centroid seeds (default 10)
        :return: (np.array) Optimal allocation using the NCO algorithm.
        """

        pass

    def allocate_mcos(self, mu_vec, cov, num_obs, num_sims=100, kde_bwidth=0.01, min_var_portf=True, lw_shrinkage=False):
        """
        Estimates the optimal allocation using the Monte Carlo optimization selection (MCOS) algorithm.

        Repeats the CVO and the NCO algorithms multiple times on the empirical values to get a dataframe of trials
        in order to later compare them to a true optimal weights allocation and compare the robustness of the NCO
        and CVO methods.

        :param mu_vec: (np.array) The original vector of expected outcomes.
        :param cov: (np.array )The original covariance matrix of outcomes.
        :param num_obs: (int) The number of observations T used to compute mu_vec and cov.
        :param num_sims: (int) The number of Monte Carlo simulations to run. (100 by default)
        :param kde_bwidth: (float) The bandwidth of the KDE used to de-noise the covariance matrix. (0.01 by default)
        :param min_var_portf: (bool) When True, the minimum variance solution is computed. Otherwise, the
                                     maximum Sharpe ratio solution is computed. (True by default)
        :param lw_shrinkage: (bool) When True, the covariance matrix is subjected to the Ledoit-Wolf shrinkage
                                    procedure. (False by default)
        :return: (pd.DataFrame, pd.DataFrame) DataFrames with allocations for CVO and NCO algorithms.
        """

        pass

    def estim_errors_mcos(self, w_cvo, w_nco, mu_vec, cov, min_var_portf=True):
        """
        Computes the true optimal allocation w, and compares that result with the estimated ones by MCOS.

        The result is the mean standard deviation between the true weights and the ones obtained from the simulation
        for each algorithm - CVO and NCO.

        :param w_cvo: (pd.DataFrame) DataFrame with weights from the CVO algorithm.
        :param w_nco: (pd.DataFrame) DataFrame with weights from the NCO algorithm.
        :param mu_vec: (np.array) The original vector of expected outcomes.
        :param cov: (np.array)The original covariance matrix of outcomes.
        :param min_var_portf: (bool) When True, the minimum variance solution was computed. Otherwise, the
                                     maximum Sharpe ratio solution was computed. (True by default)
        :return: (float, float) Mean standard deviation of weights for CVO and NCO algorithms.
        """

        pass

    @staticmethod
    def _simulate_covariance(mu_vector, cov_matrix, num_obs, lw_shrinkage=False):
        """
        Derives an empirical vector of means and an empirical covariance matrix.

        Based on the set of true means vector and covariance matrix of X distributions,
        the function generates num_obs observations for every X.
        Based on these observations simulated vector of means and the simulated covariance
        matrix are obtained.

        :param mu_vector: (np.array) True means vector for X distributions
        :param cov_matrix: (np.array) True covariance matrix for X distributions
        :param num_obs: (int) Number of observations to draw for every X
        :param lw_shrinkage: (bool) Flag to apply Ledoit-Wolf shrinkage to X (False by default)
        :return: (np.array, np.array) Empirical means vector, empirical covariance matrix
        """

        pass

    @staticmethod
    def _cluster_kmeans_base(corr, max_num_clusters=None, n_init=10):
        """
        Finding the optimal partition of clusters using K-Means algorithm.

        For the fit of K-Means algorithm a matrix of distances based on the correlation matrix is used.
        The algorithm iterates n_init number of times and initialises K-Means with different seeds
        and max_number_of_clusters.

        The Silhouette Coefficient is used as a measure of how well samples are clustered
        with samples that are similar to themselves.

        :param corr: (pd.DataFrame) DataFrame with correlation matrix
        :param max_num_clusters: (float) Maximum allowed number of clusters. If None then taken as num_elements/2
        :param n_init: (float) Number of time the k-means algorithm will run with different centroid seeds (default 10)
        :return: (np.array, dict, pd.Series) Correlation matrix of clustered elements, dict with clusters,
                                             Silhouette Coefficient series
        """

        pass

    @staticmethod
    def _form_block_matrix(num_blocks, block_size, block_corr):
        """
        Creates a correlation matrix in a block form with given parameters.

        :param num_blocks: (int) Number of blocks in matrix
        :param block_size: (int) Size of a single block
        :param block_corr: (float) Correlation of elements in a block
        :return: (np.array) Resulting correlation matrix
        """

        pass

    def form_true_matrix(self, num_blocks, block_size, block_corr, std=None):
        """
        Creates a random vector of means and a random covariance matrix.

        Due to the block structure of a matrix, it is a good sample data to use in the NCO and MCOS algorithms.

        The number of assets in a portfolio, number of blocks and correlations
        both inside the cluster and between clusters are adjustable.

        :param num_blocks: (int) Number of blocks in matrix
        :param block_size: (int) Size of a single block
        :param block_corr: (float) Correlation of elements in a block
        :param std: (float) Correlation between the clusters. If None, taken a random value from uniform dist[0.05, 0.2]
        :return: (np.array, pd.DataFrame) Resulting vector of means and the dataframe with covariance matrix
        """
        pass
