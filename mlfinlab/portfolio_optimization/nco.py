# pylint: disable=missing-module-docstring

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
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
        if len(observations.shape) == 1:  # If the input vector is one-dimentrional, reshaping
            observations = observations.reshape(-1, 1)

        # Estimating Kernel Density of the empirical distribution of eigenvalues
        kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bwidth).fit(observations)

        # If no specific values provided, the fit KDE will be avalued on unique eigenvalues.
        if eval_points is None:
            eval_points = np.unique(observations).reshape(-1, 1)

        # If the input vector is one-dimentrional, reshaping
        if len(eval_points.shape) == 1:
            eval_points = eval_points.reshape(-1, 1)

        # Evaluateing the log density model on the given values
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
