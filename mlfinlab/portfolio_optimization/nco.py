# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


class NCO:
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

        return

    @staticmethod
    def allocate_cvo(cov, mu_vec=None):
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

        # Calculating the inverse covariance matrix
        inv_cov = np.linalg.inv(cov)

        # Generating a vector of size of the inverted covariance matrix
        ones = np.ones(shape=(inv_cov.shape[0], 1))

        if mu_vec is None:  # To output the minimum variance portfolio
            mu_vec = ones

        # Calculating the analytical solution using CVO - weights
        w_cvo = np.dot(inv_cov, mu_vec)
        w_cvo /= np.dot(mu_vec.T, w_cvo)

        return w_cvo

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

        # Using pd.DataFrame instead of np.array
        cov = pd.DataFrame(cov)

        # Optimal solution for minimum variance
        if mu_vec is not None:
            mu_vec = pd.Series(mu_vec[:, 0])

        # Class with function to calculate correlation from covariance function
        risk_estimators = RiskEstimators()

        # Calculating correlation matrix
        corr = risk_estimators.cov_to_corr(cov)

        # Optimal partition of clusters (step 1)
        corr, clusters, _ = self._cluster_kmeans_base(corr, max_num_clusters, n_init=n_init)

        # Weights inside clusters
        w_intra_clusters = pd.DataFrame(0, index=cov.index, columns=clusters.keys())

        # Iterating over clusters
        for i in clusters:
            # Covariance matrix of elements in cluster
            cov_cluster = cov.loc[clusters[i], clusters[i]].values

            # Optimal solution vector for the cluster
            mu_cluster = (None if mu_vec is None else mu_vec.loc[clusters[i]].values.reshape(-1, 1))

            # Estimating the Convex Optimization Solution in a cluster (step 2)
            w_intra_clusters.loc[clusters[i], i] = self.allocate_cvo(cov_cluster, mu_cluster).flatten()

        # Reducing new covariance matrix to calculate inter-cluster weights
        cov_inter_cluster = w_intra_clusters.T.dot(np.dot(cov, w_intra_clusters))
        mu_inter_cluster = (None if mu_vec is None else w_intra_clusters.T.dot(mu_vec))

        # Optimal allocations across the reduced covariance matrix (step 3)
        w_inter_clusters = pd.Series(self.allocate_cvo(cov_inter_cluster, mu_inter_cluster).flatten(),
                                     index=cov_inter_cluster.index)

        # Final allocations - dot-product of the intra-cluster and inter-cluster allocations (step 4)
        w_nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1, 1)

        return w_nco

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
        :return: (pd.dataframe, pd.dataframe) DataFrames with allocations for CVO and NCO algorithms.
        """

        # Creating DataFrames for CVO results and NCO results
        w_cvo = pd.DataFrame(columns=range(cov.shape[0]), index=range(num_sims), dtype=float)
        w_nco = w_cvo.copy(deep=True)

        # Iterating thorough simulations
        for simulation in range(num_sims):
            # Deriving empirical vector of means and an empirical covariance matrix
            mu_simulation, cov_simulation = self._simulate_covariance(mu_vec, cov, num_obs, lw_shrinkage)

            # If goal is minimum variance
            if min_var_portf:
                mu_simulation = None

            # Class with de-noising function
            risk_estimators = RiskEstimators()

            # De-noising covariance matrix
            if kde_bwidth > 0:
                cov_simulation = risk_estimators.denoise_covariance(cov_simulation, num_obs / cov_simulation.shape[1],
                                                                    kde_bwidth)

            # Writing the results to corresponding dataframes
            w_cvo.loc[simulation] = self.allocate_cvo(cov_simulation, mu_simulation).flatten()
            w_nco.loc[simulation] = self.allocate_nco(cov_simulation, mu_simulation,
                                                      int(cov_simulation.shape[0] / 2)).flatten()

        return w_cvo, w_nco

    def estim_errors_mcos(self, w_cvo, w_nco, mu_vec, cov, min_var_portf=True):
        """
        Computes the true optimal allocation w, and compares that result with the estimated ones by MCOS.

        The result is the mean standard deviation between the true weights and the ones obtained from the simulation
        for each algorithm - CVO and NCO.

        :param w_cvo: (pd.dataframe) DataFrame with weights from the CVO algorithm.
        :param w_nco: (pd.dataframe) DataFrame with weights from the NCO algorithm.
        :param mu_vec: (np.array) The original vector of expected outcomes.
        :param cov: (np.array)The original covariance matrix of outcomes.
        :param min_var_portf: (bool) When True, the minimum variance solution was computed. Otherwise, the
                                     maximum Sharpe ratio solution was computed. (True by default)
        :return: (float, float) Mean standard deviation of weights for CVO and NCO algorithms.
        """

        # Calculating the true optimal weights allocation
        w_true = self.allocate_cvo(cov, None if min_var_portf else mu_vec)
        w_true = np.repeat(w_true.T, w_cvo.shape[0], axis=0)

        # Mean standard deviation between the weights from CVO algorithm and the true weights
        err_cvo = (w_cvo - w_true).std(axis=0).mean()

        # Mean standard deviation between the weights from NCO algorithm and the true weights
        err_nco = (w_nco - w_true).std(axis=0).mean()

        return err_cvo, err_nco

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

        # Generating a matrix of num_obs observations for X distributions
        observations = np.random.multivariate_normal(mu_vector.flatten(), cov_matrix, size=num_obs)

        # Empirical means vector calculation
        mu_simulated = observations.mean(axis=0).reshape(-1, 1)

        if lw_shrinkage:  # If applying Ledoit-Wolf shrinkage
            cov_simulated = LedoitWolf().fit(observations).covariance_

        else:  # Simple empirical covariance matrix
            cov_simulated = np.cov(observations, rowvar=False)

        return mu_simulated, cov_simulated

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

        # Distance matrix from correlation matrix
        dist_matrix = ((1 - corr.fillna(0)) / 2) ** (1 / 2)

        # Series for Silhouette Coefficients - cluster fit measure
        silh_coef_optimal = pd.Series()

        # If maximum number of clusters undefined, it's equal to half the number of elements
        if max_num_clusters is None:
            max_num_clusters = round(corr.shape[0] / 2)

        # Iterating over the allowed iteration times for k-means
        for init in range(1, n_init):
            # Iterating through every number of clusters
            for num_clusters in range(2, max_num_clusters + 1):
                # Computing k-means clustering
                kmeans = KMeans(n_clusters=num_clusters, n_jobs=1, n_init=init)
                kmeans = kmeans.fit(dist_matrix)

                # Computing a Silhouette Coefficient - cluster fit measure
                silh_coef = silhouette_samples(dist_matrix, kmeans.labels_)

                # Metrics to compare numbers of clusters
                stat = (silh_coef.mean() / silh_coef.std(), silh_coef_optimal.mean() / silh_coef_optimal.std())

                # If this is the first metric or better than the previous
                # we set it as the optimal number of clusters
                if np.isnan(stat[1]) or stat[0] > stat[1]:
                    silh_coef_optimal = silh_coef
                    kmeans_optimal = kmeans

        # Sorting labels of clusters
        new_index = np.argsort(kmeans_optimal.labels_)

        # Reordering correlation matrix rows
        corr = corr.iloc[new_index]

        # Reordering correlation matrix columns
        corr = corr.iloc[:, new_index]

        # Preparing cluster members as dict
        clusters = {i: corr.columns[np.where(kmeans_optimal.labels_ == i)[0]].tolist() for \
                    i in np.unique(kmeans_optimal.labels_)}

        # Silhouette Coefficient series
        silh_coef_optimal = pd.Series(silh_coef_optimal, index=dist_matrix.index)

        return corr, clusters, silh_coef_optimal

    @staticmethod
    def _form_block_matrix(num_blocks, block_size, block_corr):
        """
        Creates a correlation matrix in a block form with given parameters.

        :param num_blocks: (int) Number of blocks in matrix
        :param block_size: (int) Size of a single block
        :param block_corr: (float) Correlation of elements in a block
        :return: (np.array) Resulting correlation matrix
        """

        # Creating a single block with all elements as block_corr
        block = np.ones((block_size, block_size)) * block_corr

        # Setting the main diagonal to ones
        block[range(block_size), range(block_size)] = 1

        # Create a block diagonal matrix with a number of equal blocks
        res_matrix = block_diag(*([block] * num_blocks))

        return res_matrix

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
        :return: (np.array, pd.dataframe) Resulting vector of means and the dataframe with covariance matrix
        """

        # Creating a block correlation matrix
        corr_matrix = self._form_block_matrix(num_blocks, block_size, block_corr)

        # Transforming to DataFrame
        corr_matrix = pd.DataFrame(corr_matrix)

        # Getting columns of matrix separately
        columns = corr_matrix.columns.tolist()

        # Randomizing the order of the columns
        np.random.shuffle(columns)
        corr_matrix = corr_matrix[columns].loc[columns].copy(deep=True)

        if std is None:  # Default intra-cluster correlations at 0.5
            std = np.random.uniform(.05, .2, corr_matrix.shape[0])
        else:  # Or the ones set by user
            std = np.array([std] * corr_matrix.shape[1])

        # Class to calculate covariance from the correlation function
        risk_estimators = RiskEstimators()

        # Calculating covariance matrix from the generated correlation matrix
        cov_matrix = risk_estimators.corr_to_cov(corr_matrix, std)

        # Vector of means
        mu_vec = np.random.normal(std, std, cov_matrix.shape[0]).reshape(-1, 1)

        return mu_vec, cov_matrix
