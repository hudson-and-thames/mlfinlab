# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.optimize import minimize


class NCO:
    """
    This class implements the Nested Clustered Optimization (NCO) algorithm. It is reproduced with
    modification from the following paper: `Marcos Lopez de Prado “A Robust Estimator of the Efficient Frontier”,
    (2019). <https://papers.ssrn.com/abstract_id=3469961>`_.
    """

    def __init__(self):
        """
        Initialize
        """

    @staticmethod
    def simulate_covariance(mu_vector, cov_matrix, num_obs, lw_shrinkage=False):
        """
        Derives an empirical vector of means and an empirical covariance matrix.

        Based on the set of true means vector and covariance matrix of X distributions,
        the function generates num_obs observations for every X.
        Based on these observations simulated vector of means and the simulated covariance
        matrix are obtained.

        :param mu_vector: (np.array) True means vector for X distributions
        :param cov_matrix: (np.array) True covariance matrix for X distributions
        :param num_obs: (int) Number of observations to draw for every X
        :param lw_shrinkage: (bool) Flag to apply Ledoit-Wolf shrinkage to X
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
    def fit_kde(observations, kde_bwidth=0.25, kde_kernel='gaussian', eval_points=None):
        """
        Fits kernel to a series of observations, and derives the probability of observations.

        :param observations: (np.array) Array of eigenvalues to fit kernel to
        :param kde_bwidth: (float) The bandwidth of the kernel
        :param kde_kernel: (str) Kernel to use [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
        :param eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated
                                       If not provided, the unique values of observations are used
        :return: (pd.Series) Series of log(density) of the eval_points
        """

        if len(observations.shape) == 1:  # If the input vector is one-dimensional, reshaping
            observations = observations.reshape(-1, 1)

        # Estimating Kernel Density of the empirical distribution of eigenvalues
        kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bwidth).fit(observations)

        # If no specific values provided, the fit KDE will be valued on unique eigenvalues.
        if eval_points is None:
            eval_points = np.unique(observations).reshape(-1, 1)

        # If the input vector is one-dimensional, reshaping
        if len(eval_points.shape) == 1:
            eval_points = eval_points.reshape(-1, 1)

        # Evaluating the log density model on the given values
        log_prob = kde.score_samples(eval_points)

        # Preparing the output of log densities
        pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

        return pdf

    @staticmethod
    def mp_pdf(var, tn_relation, num_points):
        """
        Derives the pdf of the Marcenko-Pastur distribution.

        Outputs the pdf for num_points between the minimum and maximum expected eigenvalues.
        Requires the variance of the distribution (var) and the relation of T - the number
        of observations of each X variable to N - the number of X variables.

        :param var: (float) Variance of the distribution
        :param tn_relation: (float) Relation of sample length T to the number of variables N
        :param num_points: (int) Number of points to estimate pdf
        :return: (pd.Series) Series of pdf values
        """

        # Minimum and maximum expected eigenvalues
        eigen_min = var * (1 - (1 / tn_relation) ** (1 / 2)) ** 2
        eigen_max = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        # Space of eigenvalues
        eigen_space = np.linspace(eigen_min, eigen_max, num_points)

        # Marcenko-Pastur probability density function for eigen_space
        pdf = tn_relation * ((eigen_max - eigen_space) * (eigen_space - eigen_min)) ** (1 / 2) / (2 * np.pi * var * eigen_space)
        pdf = pd.Series(pdf, index=eigen_space)

        return pdf

    def pdf_fit(self, var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):
        """
        Calculates the fit (Sum of Squared estimate of Errors) of the empirical pdf
        (kernel density estimation) to the theoretical pdf (Marcenko-Pastur distribution).

        SSE is calculated for num_points, equally spread between minimum and maximum
        expected theoretical eigenvalues.

        :param var: (float) Variance of the distribution (theoretical pdf)
        :param eigen_observations: (np.array) Observed empirical eigenvalues (empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N (theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel (empirical pdf)
        :param num_points: (int) Number of points to estimate pdf  (empirical pdf)
        :return: (float) SSE between empirical pdf and theoretical pdf
        """

        # Calculating theoretical and empirical pdf
        theoretical_pdf = self.mp_pdf(var, tn_relation, num_points)
        empirical_pdf = self.fit_kde(eigen_observations, kde_bwidth, eval_points=theoretical_pdf.index.values)

        # Fit calculation
        sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)

        return sse

    def find_max_eval(self, eigen_observations, tn_relation, kde_bwidth):
        """
        Searching for maximum random eigenvalue by fitting Marcenko-Pastur distribution
        to the empirical one - obtained through kernel density estimation.

        :param eigen_observations: (np.array) Observed empirical eigenvalues (empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N (theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel (empirical pdf)
        :return: (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution
        """

        # Searching for the variation of Marcenko-Pastur distribution for the best fit with empirical distribution
        optimization = minimize(lambda *x: self.pdf_fit(*x), x0=np.array(0.5), args=(eigen_observations, tn_relation, kde_bwidth),
                                bounds=((1E-5, 1 - 1E-5),))

        if optimization['success']:  # If optimal variation found
            var = optimization['x'][0]
        else:  # If not found
            var = 1

        # Eigenvalue calculated as the maximum expected eigenvalue based on the input
        maximum_eigen = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        return maximum_eigen, var

    @staticmethod
    def corr_to_cov(corr, std):
        """
        Recovers the covariance matrix from the de-noise correlation matrix.

        :param corr: (np.array) Correlation matrix
        :param std: (np.array) vector of standard deviations
        :return: (np.array) Covariance matrix
        """

        cov = corr * np.outer(std, std)

        return cov

    @staticmethod
    def cov_to_corr(cov):
        """
        Derives the correlation matrix from a covariance matrix.

        :param cov: (np.array) Covariance matrix
        :return: (np.array) Covariance matrix
        """

        # Calculating standard deviations of the elements
        std = np.sqrt(np.diag(cov))

        # Transforming to correlation matrix
        corr = cov / np.outer(std, std)

        # Making sure correlation coefficients are in (-1, 1) range
        corr[corr < -1], corr[corr > 1] = -1, 1

        return corr

    @staticmethod
    def get_pca(hermit_matrix):
        """
        Calculates eigenvalues and eigenvectors from a Hermitian matrix.

        Eigenvalues in the output are on the main diagonal of a matrix.

        :param hermit_matrix: (np.array) Hermitian matrix
        :return: (np.array, np.array) Eigenvalues matrix, eigenvectors array
        """

        # Calculating eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(hermit_matrix)

        # Index to sort eigenvalues in descending order
        indices = eigenvalues.argsort()[::-1]

        # Sorting
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Outputting eigenvalues on the main diagonal of a matrix
        eigenvalues = np.diagflat(eigenvalues)

        return eigenvalues, eigenvectors

    def denoised_corr(self, eigenvalues, eigenvectors, num_facts):
        """
        Shrinks the eigenvalues associated with noise, and returns a de-noised correlation matrix

        Noise is removed from correlation matrix by fixing random eigenvalues.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal
        :param eigenvectors: (float) Eigenvectors array
        :param num_facts: (float) Threshold for eigenvalues to be fixed
        :return: (np.array) De-noised correlation matrix
        """

        # Vector of eigenvalues from main diagonal of a matrix
        eigenval_vec = np.diag(eigenvalues).copy()

        # Replacing eigenvalues after num_facts to their average value
        eigenval_vec[num_facts:] = eigenval_vec[num_facts:].sum() / float(eigenval_vec.shape[0] - num_facts)

        # Back to eigenvalues on main diagonal of a matrix
        eigenvalues = np.diag(eigenval_vec)

        # De-noised covariance matrix
        cov = np.dot(eigenvectors, eigenvalues).dot(eigenvectors.T)

        # Ne-noised correlation matrix
        corr = self.cov_to_corr(cov)

        return corr

    def de_noise_cov(self, cov, tn_relation, kde_bwidth):
        """
        Computes a denoised covariation matrix from a given covariation matrix.

        As a threshold for the denoising the correlation matrix, the maximum eigenvalue
        that fits the theoretical distribution is used.

        :param cov: (np.array) Covariance matrix
        :param tn_relation: (float) Relation of sample length T to the number of variables N
        :param kde_bwidth: (float) The bandwidth of the kernel
        :return: (np.array) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution
        """

        # Correlation matrix computation
        corr = self.cov_to_corr(cov)

        # Calculating eigenvalues and eigenvectors
        eigenval, eigenvec = self.get_pca(corr)

        # Calculating the maximum eigenvalue to fit the theoretical distribution
        maximum_eigen, _ = self.find_max_eval(np.diag(eigenval), tn_relation, kde_bwidth)

        # Calculating the threshold of eigenvalues that fit theoretical distribution
        # from our set of eigenvalues
        num_facts = eigenval.shape[0] - np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

        # Based on the threshold, de-noising the correlation matrix
        corr = self.denoised_corr(eigenval, eigenvec, num_facts)

        # Calculating the covariance matrix
        cov_denoised = self.corr_to_cov(corr, np.diag(cov) ** (1 / 2))

        return cov_denoised

    @staticmethod
    def cluster_kmeans_base(corr, max_num_clusters=None, n_init=10):
        """
        Finding the optimal partition of clusters using K-Means algorithm.

        For the fit of K-Means algorithm a matrix of distances based on the correlation matrix is used.
        The algorithm iterates through n_init - time K-Means can run with different seeds and
        max_num_clusters - maximum number of clusters allowed.

        The Silhouette Coefficient is used as a measure of how well samples are clustered
        with samples that are similar to themselves.

        :param corr: (np.array) Correlation matrix
        :param max_num_clusters: (float) Maximum allowed number of clusters
        :param n_init: (float) Number of time the k-means algorithm will be run with different centroid seeds
        :return: (np.array, dict, pd.Series) Correlation matrix of clustered elements, dict with clusters,
                                             Silhouette Coefficient series
        """

        # Distance matix from correlation matrix
        dist_matrix = ((1 - corr.fillna(0)) / 2) ** (1 / 2)

        # Series for Silhouette Coefficients - cluster fit measure
        silh_coef_optimal = pd.Series()

        # If maximum number of clusters undefined, it's equal to the half of the elements
        if max_num_clusters is None:
            max_num_clusters = corr.shape[0] / 2

        # Iterating over the allowed iteration times for k-means
        for init in range(n_init):
            # Iterating through every number of clusters
            for i in range(2, max_num_clusters + 1):
                # Computing k-means clustering
                kmeans = KMeans(n_clusters=i, n_jobs=1, n_init=init)
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
    def opt_port(cov, mu_vec=None):
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
        :return: (np.array) Optimal allocation
        """

        # Calculating the inverse covariance matrix
        inv_cov = np.linalg.inv(cov)

        # Generating a vector of size of the inverted covariance matrix
        ones = np.ones(shape=(inv_cov.shape[0], 1))

        if mu_vec is None:  # To output the minimum variance portfolio
            mu_vec = ones

        # Calculating the analytical solution - weights
        w_cvo = np.dot(inv_cov, mu_vec)
        w_cvo /= np.dot(mu_vec.T, w_cvo)

        return w_cvo


    def opt_port_nco(self, cov, mu_vec=None, max_num_clusters=None):
        """
        Estimates the optimal allocation using the nested clustered optimization (NCO) algorithm.

        First, it clusters the covariance matrix into subsets of highly correlated variables.
        Second, it computes the optimal allocation for each of the clusters separately.
        This allows to collapse the original covariance matrix into a reduced covariance matrix,
        where each cluster is represented by a single variable.
        Third, we compute the optimal allocations across the reduced covariance matrix.
        Fourth, the final allocations are the dot-product of the intra-cluster (step 2) allocations and
        the inter-cluster (step 3) allocations.

        For the Convex Optimization Solution, a mu - optimal solution parameter is needed.
        If mu is the vector of expected values from variables, the result will be
        a vector of weights with maximum Sharpe ratio.
        If mu is a vector of ones (pass None value), the result will be a vector of weights with
        minimum variance.

        :param cov: (np.array) Covariance matrix of the variables.
        :param mu_vec: (np.array) Expected value of draws from the variables for maximum Sharpe ratio.
                              None if outputting the minimum variance portfolio.
        :param max_тum_сlusters: (int) Allowed maximum number of clusters.
        :return: (np.array) Optimal allocation using the NCO algorithm.
        """

        # Using pd.DataFrame instead of np.array
        cov = pd.DataFrame(cov)

        # Optimal solution for minimum variance
        if mu_vec is not None:
            mu_vec = pd.Series(mu_vec[:, 0])

        # Calculating correlation matrix
        corr = self.cov_to_corr(cov)

        # Optimal partition of clusters (step 1)
        corr, clusters, _ = self.cluster_kmeans_base(corr, max_num_clusters, n_init=10)

        # Weights inside clusters
        w_intra_clusters = pd.DataFrame(0, index=cov.index, columns=clusters.keys())

        # Iterating over clusters
        for i in clusters:
            # Covariance matrix of elements in cluster
            cov_cluster = cov.loc[clusters[i], clusters[i]].values

            # Optimal solution vector for the cluster
            mu_cluster = (None if mu_vec is None else mu_vec.loc[clusters[i]].values.reshape(-1, 1))

            # Estimating the Convex Optimization Solution in a cluster (step 2)
            w_intra_clusters.loc[clusters[i], i] = self.opt_port(cov_cluster, mu_cluster).flatten()

        # Reducing new covariance matrix to calculate inter-cluster weights
        cov_inter_cluster = w_intra_clusters.T.dot(np.dot(cov, w_intra_clusters))
        mu_inter_cluster = (None if mu_vec is None else w_intra_clusters.T.dot(mu_vec))

        # Optimal allocations across the reduced covariance matrix (step 3)
        w_inter_clusters = pd.Series(self.opt_port(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)

        # Final allocations - dot-product of the intra-cluster and inter-cluster allocations (step 4)
        w_nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1, 1)

        return w_nco
