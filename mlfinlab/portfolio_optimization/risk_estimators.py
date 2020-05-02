# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.covariance import MinCovDet, EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS
from scipy.optimize import minimize
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class RiskEstimators:
    """
    This class contains the implementations for different ways to calculate and adjust Covariance matrices.
    The functions related to de-noising the Covariance matrix are reproduced with modification from the following paper:
    `Marcos Lopez de Prado “A Robust Estimator of the Efficient Frontier”, (2019).
    <https://papers.ssrn.com/abstract_id=3469961>`_.
    """

    def __init__(self):
        """
        Initialize
        """

        return

    @staticmethod
    def _fit_kde(observations, kde_bwidth=0.01, kde_kernel='gaussian', eval_points=None):
        """
        Fits kernel to a series of observations (in out case eigenvalues), and derives the
        probability density function of observations.

        :param observations: (np.array) Array of observations (eigenvalues) eigenvalues to fit kernel to
        :param kde_bwidth: (float) The bandwidth of the kernel
        :param kde_kernel: (str) Kernel to use [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
        :param eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated.
                                       If None, the unique values of observations are used
        :return: (pd.Series) Series with estimated pdf values in the eval_points
        """

        # Reshaping array to a vertical one
        observations = observations.reshape(-1, 1)

        # Estimating Kernel Density of the empirical distribution of eigenvalues
        kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bwidth).fit(observations)

        # If no specific values provided, the fit KDE will be valued on unique eigenvalues.
        if eval_points is None:
            eval_points = np.unique(observations).reshape(-1, 1)

        # If the input vector is one-dimensional, reshaping to a vertical one
        if len(eval_points.shape) == 1:
            eval_points = eval_points.reshape(-1, 1)

        # Evaluating the log density model on the given values
        log_prob = kde.score_samples(eval_points)

        # Preparing the output of pdf values
        pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

        return pdf

    @staticmethod
    def _mp_pdf(var, tn_relation, num_points):
        """
        Derives the pdf of the Marcenko-Pastur distribution.

        Outputs the pdf for num_points between the minimum and maximum expected eigenvalues.
        Requires the variance of the distribution (var) and the relation of T - the number
        of observations of each X variable to N - the number of X variables (T/N).

        :param var: (float) Variance of the M-P distribution
        :param tn_relation: (float) Relation of sample length T to the number of variables N (T/N)
        :param num_points: (int) Number of points to estimate pdf
        :return: (pd.Series) Series of M-P pdf values
        """

        # Changing the type as scipy.optimize.minimize outputs np.array with one element to this function
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

        :param var: (float) Variance of the M-P distribution (for the theoretical pdf)
        :param eigen_observations: (np.array) Observed empirical eigenvalues (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel (for the empirical pdf)
        :param num_points: (int) Number of points to estimate pdf  (for the empirical pdf)
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

        :param eigen_observations: (np.array) Observed empirical eigenvalues (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel (for the empirical pdf)
        :return: (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution
        """

        # Searching for the variation of Marcenko-Pastur distribution for the best fit with the empirical distribution
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
        Recovers the covariance matrix from a correlation matrix.

        :param corr: (np.array) Correlation matrix
        :param std: (np.array) Vector of standard deviations
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
        Calculates eigenvalues and eigenvectors from a Hermitian matrix. In our case, from the correlation matrix.

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
        Shrinks the eigenvalues associated with noise, and returns a de-noised correlation matrix.

        Noise is removed from the correlation matrix by fixing random eigenvalues.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal
        :param eigenvectors: (float) Eigenvectors array
        :param num_facts: (float) Threshold for eigenvalues to be fixed
        :return: (np.array) De-noised correlation matrix
        """

        # Vector of eigenvalues from the main diagonal of a matrix
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

    def denoise_covariance(self, cov, tn_relation, kde_bwidth=0.01):
        """
        Computes a de-noised covariance/correlation matrix from a given covariance/correlation matrix.

        As a threshold for the denoising the correlation matrix, the maximum eigenvalue
        that fits the theoretical distribution is used.

        This algorithm is reproduced with minor modifications from the following paper:
        `Marcos Lopez de Prado “A Robust Estimator of the Efficient Frontier”, (2019).
        <https://papers.ssrn.com/abstract_id=3469961>`_.

        :param cov: (np.array) Covariance/correlation matrix
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    covariance/correlation matrix.
        :param kde_bwidth: (float) The bandwidth of the kernel to fit
        :return: (np.array) De-noised covariance/correlation matrix
        """

        # Correlation matrix computation (if correlation matrix given, nothing changes)
        corr = self.cov_to_corr(cov)

        # Calculating eigenvalues and eigenvectors
        eigenval, eigenvec = self._get_pca(corr)

        # Calculating the maximum eigenvalue to fit the theoretical distribution
        maximum_eigen, _ = self._find_max_eval(np.diag(eigenval), tn_relation, kde_bwidth)

        # Calculating the threshold of eigenvalues that fit the theoretical distribution
        # from our set of eigenvalues
        num_facts = eigenval.shape[0] - np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

        # Based on the threshold, de-noising the correlation matrix
        corr = self._denoised_corr(eigenval, eigenvec, num_facts)

        # Calculating the covariance matrix from the de-noised correlation matrix
        cov_denoised = self.corr_to_cov(corr, np.diag(cov) ** (1 / 2))

        return cov_denoised

    @staticmethod
    def minimum_covariance_determinant(returns, price_data=False, assume_centered=False,
                                       support_fraction=None, random_state=None):
        """
        Calculates the Minimum Covariance Determinant for a dataframe of asset prices or returns.

        This function is a wrap of the sklearn's MinCovDet (MCD) class. According to the
        scikit-learn User Guide on Covariance estimation:

        "The idea is to find a given proportion (h) of “good” observations that are not outliers
        and compute their empirical covariance matrix. This empirical covariance matrix is then
        rescaled to compensate for the performed selection of observations".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimation class.

        :param returns: (pd.dataframe) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns.
        :param assume_centered: (bool) Flag for data with mean significantly equal to zero
                                       (Read the documentation for MinCovDet class).
        :param support_fraction: (float) Values between 0 and 1. The proportion of points to be included in the support
                                         of the raw MCD estimate (Read the documentation for MinCovDet class).
        :param random_state: (int) Seed used by the random number generator.
        :return: (np.array) Estimated robust covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimation()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Calculating the covariance matrix
        cov_matrix = MinCovDet(assume_centered=assume_centered, support_fraction=support_fraction,
                               random_state=random_state).fit(returns).covariance_

        return cov_matrix

    @staticmethod
    def empirical_covariance(returns, price_data=False, assume_centered=False):
        """
        Calculates the Maximum likelihood covariance estimator for a dataframe of asset prices or returns.

        This function is a wrap of the sklearn's EmpiricalCovariance class. According to the
        scikit-learn User Guide on Covariance estimation:

        "The covariance matrix of a data set is known to be well approximated by the classical maximum
        likelihood estimator, provided the number of observations is large enough compared to the number
        of features (the variables describing the observations). More precisely, the Maximum Likelihood
        Estimator of a sample is an unbiased estimator of the corresponding population’s covariance matrix".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimation class.

        :param returns: (pd.dataframe) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns.
        :param assume_centered: (bool) Flag for data with mean almost, but not exactly zero
                                       (Read documentation for EmpiricalCovariance class).
        :return: (np.array) Estimated covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimation()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Calculating the covariance matrix
        cov_matrix = EmpiricalCovariance(assume_centered=assume_centered).fit(returns).covariance_

        return cov_matrix

    @staticmethod
    def shrinked_covariance(returns, price_data=False, shrinkage_type='basic', assume_centered=False,
                            basic_shrinkage=0.1):
        """
        Calculates the Covariance estimator with shrinkage for a dataframe of asset prices or returns.

        This function allows three types of shrinkage - Basic, Ledoit-Wolf and Oracle Approximating Shrinkage.
        It is a wrap of the sklearn's ShrunkCovariance, LedoitWolf and OAS classes. According to the
        scikit-learn User Guide on Covariance estimation:

        "Sometimes, it even occurs that the empirical covariance matrix cannot be inverted for numerical
        reasons. To avoid such an inversion problem, a transformation of the empirical covariance matrix
        has been introduced: the shrinkage. Mathematically, this shrinkage consists in reducing the ratio
        between the smallest and the largest eigenvalues of the empirical covariance matrix".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/covariance.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimation class.

        :param returns: (pd.dataframe) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns.
        :param shrinkage_type: (str) Type of shrinkage to use ('basic','lw','oas','all').
        :param assume_centered: (bool) Flag for data with mean almost, but not exactly zero
                                       (Read documentation for chosen shrinkage class).
        :param basic_shrinkage: (float) Between 0 and 1. Coefficient in the convex combination for basic shrinkage.
        :return: (np.array) Estimated covariance matrix. Tuple of covariance matrices if shrinkage_type = 'all'.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimation()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Calculating the covariance matrix for the chosen method
        if shrinkage_type == 'basic':
            cov_matrix = ShrunkCovariance(assume_centered=assume_centered, shrinkage=basic_shrinkage).fit(
                returns).covariance_
        elif shrinkage_type == 'lw':
            cov_matrix = LedoitWolf(assume_centered=assume_centered).fit(returns).covariance_
        elif shrinkage_type == 'oas':
            cov_matrix = OAS(assume_centered=assume_centered).fit(returns).covariance_
        else:
            cov_matrix = (
                ShrunkCovariance(assume_centered=assume_centered, shrinkage=basic_shrinkage).fit(returns).covariance_,
                LedoitWolf(assume_centered=assume_centered).fit(returns).covariance_,
                OAS(assume_centered=assume_centered).fit(returns).covariance_)

        return cov_matrix

    @staticmethod
    def semi_covariance(returns, price_data=False, threshold_return=0):
        """
        Calculates the Semi-Covariance matrix for a dataframe of asset prices or returns.

        Semi-Covariance matrix is used to calculate the portfolio's downside volatility. Usually, the
        threshold return is zero and the negative volatility is measured. A threshold can be a positive number
        when one assumes a required return rate. If the threshold is above zero, the output is the volatility
        measure for returns below this threshold.

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimation class.

        :param returns: (pd.dataframe) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns.
        :param threshold_return: (float) Required return for each period in the frequency of the input data
                                         (If the input data is daily, it's a daily threshold return).
        :return: (np.array) Semi-Covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimation()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Returns that are lower than the threshold
        lower_returns = returns - threshold_return < 0

        # Calculating the minimum of 0 and returns minus threshold
        min_returns = (returns - threshold_return) * lower_returns

        # Simple covariance matrix
        semi_covariance = returns.cov()

        # Iterating to fill elements
        for row_number in range(semi_covariance.shape[0]):
            for column_number in range(semi_covariance.shape[1]):
                # Series of returns for the element from the row and column
                row_asset = min_returns.iloc[:, row_number]
                column_asset = min_returns.iloc[:, column_number]

                # Series of element-wise products
                covariance_series = row_asset * column_asset

                # Element of the Semi-Covariance matrix
                semi_cov_element = covariance_series.sum() / min_returns.size

                # Inserting the element in the Semi-Covariance matrix
                semi_covariance.iloc[row_number, column_number] = semi_cov_element

        return semi_covariance

    @staticmethod
    def exponential_covariance(returns, price_data=False, window_span=60):
        """
        Calculates the Exponentially-weighted Covariance matrix for a dataframe of asset prices or returns.

        It calculates the series of covariances between elements and then gets the last value of exponentially
        weighted moving average series from covariance series as an element in matrix.

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimation class.

        :param returns: (pd.dataframe) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns.
        :param window_span: (int) Used to specify decay in terms of span for the exponentially-weighted series.
        :return: (np.array) Exponentially-weighted Covariance matrix.
        """

        # Calculating the series of returns from series of prices
        if price_data:
            # Class with returns calculation function
            ret_est = ReturnsEstimation()

            # Calculating returns
            returns = ret_est.calculate_returns(returns)

        # Simple covariance matrix
        cov_matrix = returns.cov()

        # Iterating to fill elements
        for row_number in range(cov_matrix.shape[0]):
            for column_number in range(cov_matrix.shape[1]):
                # Series of returns for the element from the row and column
                row_asset = returns.iloc[:, row_number]
                column_asset = returns.iloc[:, column_number]

                # Series of covariance
                covariance_series = (row_asset - row_asset.mean()) * (column_asset - column_asset.mean())

                # Exponentially weighted moving average series
                ew_ma = covariance_series.ewm(span=window_span).mean()

                # Using the most current element as the Exponential Covariance value
                cov_matrix.iloc[row_number, column_number] = ew_ma[-1]

        return cov_matrix
