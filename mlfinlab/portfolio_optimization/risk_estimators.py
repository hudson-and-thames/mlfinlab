# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize


class RiskEstimators:
    """
    This class implements the functions related to de-noising of the covariance matrix. It is reproduced with
    modification from the following paper: `Marcos Lopez de Prado “A Robust Estimator of the Efficient Frontier”,
    (2019). <https://papers.ssrn.com/abstract_id=3469961>`_.
    """

    def __init__(self):
        """
        Initialize
        """

    @staticmethod
    def _fit_kde(observations, kde_bwidth=0.25, kde_kernel='gaussian', eval_points=None):
        """
        Fits kernel to a series of observations, and derives the probability of observations.

        :param observations: (np.array) Array of eigenvalues to fit kernel to
        :param kde_bwidth: (float) The bandwidth of the kernel
        :param kde_kernel: (str) Kernel to use [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
        :param eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated
                                       If not provided, the unique values of observations are used
        :return: (pd.Series) Series of log(density) of the eval_points
        """

        # Reshaping array to a horizontal one
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
    def _mp_pdf(var, tn_relation, num_points):
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

        # Changing the type as scipy.optimize.minimize outputs np.array
        if not isinstance(var, float):
            var = float(var)

        # Minimum and maximum expected eigenvalues
        eigen_min = var * (1 - (1 / tn_relation) ** (1 / 2)) ** 2
        eigen_max = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

        # Space of eigenvalues
        eigen_space = np.linspace(eigen_min, eigen_max, num_points)

        # Marcenko-Pastur probability density function for eigen_space
        pdf = tn_relation * ((eigen_max - eigen_space) * (eigen_space - eigen_min)) ** (1 / 2) / \
                             (2 * np.pi * var * eigen_space)
        pdf = pd.Series(pdf, index=eigen_space)

        return pdf

    def _pdf_fit(self, var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):
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
        theoretical_pdf = self._mp_pdf(var, tn_relation, num_points)
        empirical_pdf = self._fit_kde(eigen_observations, kde_bwidth, eval_points=theoretical_pdf.index.values)

        # Fit calculation
        sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)

        return sse

    def _find_max_eval(self, eigen_observations, tn_relation, kde_bwidth):
        """
        Searching for maximum random eigenvalue by fitting Marcenko-Pastur distribution
        to the empirical one - obtained through kernel density estimation.

        :param eigen_observations: (np.array) Observed empirical eigenvalues (empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N (theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel (empirical pdf)
        :return: (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution
        """

        # Searching for the variation of Marcenko-Pastur distribution for the best fit with empirical distribution
        optimization = minimize(self._pdf_fit, x0=np.array(0.5), args=(eigen_observations, tn_relation, kde_bwidth),
                                bounds=((1e-5, 1 - 1e-5),))

        # The optimal solution found
        var = optimization['x'][0]

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
    def _get_pca(hermit_matrix):
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

    def _denoised_corr(self, eigenvalues, eigenvectors, num_facts):
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

    def de_noised_cov(self, cov, tn_relation, kde_bwidth):
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
        eigenval, eigenvec = self._get_pca(corr)

        # Calculating the maximum eigenvalue to fit the theoretical distribution
        maximum_eigen, _ = self._find_max_eval(np.diag(eigenval), tn_relation, kde_bwidth)

        # Calculating the threshold of eigenvalues that fit theoretical distribution
        # from our set of eigenvalues
        num_facts = eigenval.shape[0] - np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

        # Based on the threshold, de-noising the correlation matrix
        corr = self._denoised_corr(eigenval, eigenvec, num_facts)

        # Calculating the covariance matrix
        cov_denoised = self.corr_to_cov(corr, np.diag(cov) ** (1 / 2))

        return cov_denoised
