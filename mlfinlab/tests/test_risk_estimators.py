# pylint: disable=protected-access
"""
Tests the functions from the RiskEstimators class.
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class TestRiskEstimators(unittest.TestCase):
    """
    Tests different functions of the RiskEstimators class.
    """

    def setUp(self):
        """
        Initialize and get the test data
        """

        # Stock prices data to test the Covariance functions
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

        # And series of returns
        ret_est = ReturnsEstimation()
        self.returns = ret_est.calculate_returns(self.data)

    def test_mp_pdf(self):
        """
        Test the deriving of pdf of the Marcenko-Pastur distribution.
        """

        risk_estimators = RiskEstimators()

        # Properties for the distribution
        var = 0.1
        tn_relation = 5
        num_points = 5

        # Calculating the pdf in 5 points
        pdf_mp = risk_estimators._mp_pdf(var, tn_relation, num_points)

        # Testing the minimum and maximum and non-zero values of the pdf
        self.assertAlmostEqual(pdf_mp.index[0], 0.03056, delta=1e-4)
        self.assertAlmostEqual(pdf_mp.index[4], 0.20944, delta=1e-4)

        # Testing that the distribution curve is right
        self.assertTrue(pdf_mp.values[1] > pdf_mp.values[2] > pdf_mp.values[3])

    def test_fit_kde(self):
        """
        Test the kernel fitting to a series of observations.
        """

        risk_estimators = RiskEstimators()

        # Values to fit kernel to and evaluation points
        observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        eval_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Calculating the pdf in 7 chosen points
        pdf_kde = risk_estimators._fit_kde(observations, eval_points=eval_points, kde_bwidth=0.25)

        # Testing the values and if the pdf is symmetric
        self.assertEqual(pdf_kde[0.0], pdf_kde[0.6])
        self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
        self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
        self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-5)

        # Testing also on unique values of the set as a default output
        pdf_kde_default = risk_estimators._fit_kde(observations, kde_bwidth=0.25)
        self.assertEqual(pdf_kde[0.1], pdf_kde_default[0.1])
        self.assertEqual(pdf_kde_default[0.2], pdf_kde_default[0.4])

    def test_pdf_fit(self):
        """
        Test the fit between empirical pdf and the theoretical pdf.
        """

        risk_estimators = RiskEstimators()

        # Values to calculate theoretical and empirical pdfs
        var = 0.6
        eigen_observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        tn_relation = 2
        kde_bwidth = 0.4

        # Calculating the SSE
        pdf_kde = risk_estimators._pdf_fit(var, eigen_observations, tn_relation, kde_bwidth)

        # Testing the SSE value
        self.assertAlmostEqual(pdf_kde, 50.51326, delta=1e-5)

    def test_find_max_eval(self):
        """
        Test the search for maximum random eigenvalue.
        """

        risk_estimators = RiskEstimators()

        # Values to calculate theoretical and empirical pdfs
        eigen_observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        tn_relation = 2
        kde_bwidth = 0.4

        # Optimizing and getting the maximum random eigenvalue and the optimal variation
        maximum_eigen, var = risk_estimators._find_max_eval(eigen_observations, tn_relation, kde_bwidth)

        # Testing the maximum random eigenvalue and the optimal variation
        self.assertAlmostEqual(maximum_eigen, 2.41011, delta=1e-5)
        self.assertAlmostEqual(var, 0.82702, delta=1e-5)

    @staticmethod
    def test_corr_to_cov():
        """
        Test the recovering of the covariance matrix from the correlation matrix.
        """

        risk_estimators = RiskEstimators()

        # Correlation matrix and the vector of standard deviations
        corr_matrix = np.array([[1, 0.1, -0.1],
                                [0.1, 1, -0.3],
                                [-0.1, -0.3, 1]])
        std_vec = np.array([0.1, 0.2, 0.1])

        # Expected covariance matrix
        expected_matrix = np.array([[0.01, 0.002, -0.001],
                                    [0.002, 0.04, -0.006],
                                    [-0.001, -0.006, 0.01]])

        # Finding the covariance matrix
        cov_matrix = risk_estimators.corr_to_cov(corr_matrix, std_vec)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(cov_matrix, expected_matrix, decimal=5)

    @staticmethod
    def test_cov_to_corr():
        """
        Test the deriving of the correlation matrix from a covariance matrix.
        """

        risk_estimators = RiskEstimators()

        # Covariance matrix
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])

        # Expected correlation matrix
        expected_matrix = np.array([[1, 0.1, -0.1],
                                    [0.1, 1, -0.3],
                                    [-0.1, -0.3, 1]])

        # Finding the covariance matrix
        corr_matrix = risk_estimators.cov_to_corr(cov_matrix)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(corr_matrix, expected_matrix, decimal=5)

    @staticmethod
    def test_get_pca():
        """
        Test the calculation of eigenvalues and eigenvectors from a Hermitian matrix.
        """

        risk_estimators = RiskEstimators()

        # Correlation matrix as an input
        corr_matrix = np.array([[1, 0.1, -0.1],
                                [0.1, 1, -0.3],
                                [-0.1, -0.3, 1]])

        # Expected correlation matrix
        expected_eigenvalues = np.array([[1.3562, 0, 0],
                                         [0, 0.9438, 0],
                                         [0, 0, 0.7]])
        first_eigenvector = np.array([-3.69048184e-01, -9.29410263e-01, 1.10397126e-16])

        # Finding the eigenvalues
        eigenvalues, eigenvectors = risk_estimators._get_pca(corr_matrix)

        # Testing eigenvalues and the first eigenvector
        np.testing.assert_almost_equal(eigenvalues, expected_eigenvalues, decimal=4)
        np.testing.assert_almost_equal(eigenvectors[0], first_eigenvector, decimal=5)

    @staticmethod
    def test_denoised_corr():
        """
        Test the shrinkage the eigenvalues associated with noise.
        """

        risk_estimators = RiskEstimators()

        # Eigenvalues and eigenvectors to use
        eigenvalues = np.array([[1.3562, 0, 0],
                                [0, 0.9438, 0],
                                [0, 0, 0.7]])
        eigenvectors = np.array([[-3.69048184e-01, -9.29410263e-01, 1.10397126e-16],
                                 [-6.57192300e-01, 2.60956474e-01, 7.07106781e-01],
                                 [6.57192300e-01, -2.60956474e-01, 7.07106781e-01]])

        # Expected correlation matrix
        expected_corr = np.array([[1, 0.13353165, -0.13353165],
                                  [0.13353165, 1, -0.21921986],
                                  [-0.13353165, -0.21921986, 1]])

        # Finding the eigenvalues
        corr_matrix = risk_estimators._denoised_corr(eigenvalues, eigenvectors, 1)

        # Testing if the de-noised correlation matrix is right
        np.testing.assert_almost_equal(corr_matrix, expected_corr, decimal=4)

    @staticmethod
    def test_denoise_covariance():
        """
        Test the shrinkage the eigenvalues associated with noise.
        """

        risk_estimators = RiskEstimators()

        # Covariance matrix to de-noise and parameters for the theoretical distribution.
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])
        tn_relation = 50
        kde_bwidth = 0.25

        # Expected de-noised covariance matrix
        expected_cov = np.array([[0.01, 0.00267029, -0.00133514],
                                 [0.00267029, 0.04, -0.00438387],
                                 [-0.00133514, -0.00438387, 0.01]])

        # Finding the de-noised covariance matrix
        cov_matrix_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation, kde_bwidth)

        # Testing if the de-noised covariance matrix is right
        np.testing.assert_almost_equal(cov_matrix_denoised, expected_cov, decimal=4)

    def test_minimum_covariance_determinant(self):
        """
        Test the calculation of the Minimum Covariance Determinant.
        """

        risk_estimators = RiskEstimators()

        # Getting first three columns of data to be able to compare the output
        prices_dataframe = self.data.iloc[:, :3]
        returns_dataframe = self.returns.iloc[:, :3]

        # Expected resulting Minimum Covariance Determinant
        expected_cov = np.array([[1.5110e-04, 1.1322e-04, -5.2053e-06],
                                 [1.1322e-04, 1.4760e-06, -6.6961e-06],
                                 [-5.2053e-06, -6.6961e-06, 1.0874e-05]])

        # Using the Minimum Covariance Determinant algorithm on price data with random seed 0
        min_covar_determ = risk_estimators.minimum_covariance_determinant(prices_dataframe, price_data=True,
                                                                          random_state=0)

        # Using the Minimum Covariance Determinant algorithm on return data with random seed 0
        min_covar_determ_ret = risk_estimators.minimum_covariance_determinant(returns_dataframe, price_data=False,
                                                                              random_state=0)

        # Testing if the resulting covariance matrix is right
        np.testing.assert_almost_equal(min_covar_determ, expected_cov, decimal=4)

        # And if the results for price and returns are the same
        np.testing.assert_almost_equal(min_covar_determ, min_covar_determ_ret, decimal=4)

    def test_empirical_covariance(self):
        """
        Test the calculation of the Maximum likelihood covariance estimator.
        """

        risk_estimators = RiskEstimators()

        # Getting first three columns of data to be able to compare the output
        prices_dataframe = self.data.iloc[:, :3]
        returns_dataframe = self.returns.iloc[:, :3]

        # Expected resulting Maximum likelihood covariance estimator
        expected_cov = np.array([[4.6571e-04, 3.4963e-04, -1.6626e-05],
                                 [3.4963e-04, 3.7193e-04, -1.4957e-05],
                                 [-1.6626e-05, -1.4957e-05, 1.9237e-05]])

        # Using the Maximum likelihood covariance estimator on price data
        empirical_cov = risk_estimators.empirical_covariance(prices_dataframe, price_data=True)

        # Using the Maximum likelihood covariance estimator on returns data
        empirical_cov_ret = risk_estimators.empirical_covariance(returns_dataframe, price_data=False)

        # Testing if the resulting covariance matrix is right
        np.testing.assert_almost_equal(empirical_cov, expected_cov, decimal=6)

        # And if the results for price and returns are the same
        np.testing.assert_almost_equal(empirical_cov, empirical_cov_ret, decimal=4)

    def test_shrinked_covariance(self):
        """
        Test the calculation of the Covariance estimator with shrinkage.
        """

        risk_estimators = RiskEstimators()

        # Getting first three columns of data to be able to compare the output
        prices_dataframe = self.data.iloc[:, :3]
        returns_dataframe = self.returns.iloc[:, :3]

        # Expected resulting Covariance estimators for each shrinkage type
        expected_cov_basic = np.array([[4.47705356e-04, 3.14668132e-04, -1.49635474e-05],
                                       [3.14668132e-04, 3.63299625e-04, -1.34611717e-05],
                                       [-1.49635474e-05, -1.34611717e-05, 4.58764444e-05]])

        expected_cov_lw = np.array([[4.63253312e-04, 3.44853842e-04, -1.63989814e-05],
                                    [3.44853842e-04, 3.70750646e-04, -1.47524847e-05],
                                    [-1.63989814e-05, -1.47524847e-05, 2.28774674e-05]])

        expected_cov_oas = np.array([[4.65398835e-04, 3.49019287e-04, -1.65970625e-05],
                                     [3.49019287e-04, 3.71778842e-04, -1.49306780e-05],
                                     [-1.65970625e-05, -1.49306780e-05, 1.97037481e-05]])

        # Using the Covariance estimator with different types of shrinkage on price data
        shrinked_cov_basic = risk_estimators.shrinked_covariance(prices_dataframe, price_data=True,
                                                                 shrinkage_type='basic', basic_shrinkage=0.1)

        shrinked_cov_lw = risk_estimators.shrinked_covariance(prices_dataframe, price_data=True, shrinkage_type='lw')

        shrinked_cov_oas = risk_estimators.shrinked_covariance(prices_dataframe, price_data=True, shrinkage_type='oas')

        shrinked_cov_all = risk_estimators.shrinked_covariance(prices_dataframe, price_data=True,
                                                               shrinkage_type='all', basic_shrinkage=0.1)

        # Using the Covariance estimator with different types of shrinkage on returns data
        shrinked_cov_basic_ret = risk_estimators.shrinked_covariance(returns_dataframe, price_data=False,
                                                                     shrinkage_type='basic', basic_shrinkage=0.1)

        # Testing if the resulting shrinked covariance matrix is right for every method is right
        np.testing.assert_almost_equal(shrinked_cov_basic, expected_cov_basic, decimal=7)
        np.testing.assert_almost_equal(shrinked_cov_lw, expected_cov_lw, decimal=7)
        np.testing.assert_almost_equal(shrinked_cov_oas, expected_cov_oas, decimal=7)

        # And that the results from all methods match the individual methods results
        np.testing.assert_almost_equal(shrinked_cov_all[0], shrinked_cov_basic, decimal=7)
        np.testing.assert_almost_equal(shrinked_cov_all[1], shrinked_cov_lw, decimal=7)
        np.testing.assert_almost_equal(shrinked_cov_all[2], shrinked_cov_oas, decimal=7)

        # And if the results for price and returns are the same
        np.testing.assert_almost_equal(shrinked_cov_basic, shrinked_cov_basic_ret, decimal=4)

    def test_semi_covariance(self):
        """
        Test the calculation of the Semi-Covariance matrix.
        """

        risk_estimators = RiskEstimators()

        # Getting first three columns of data to be able to compare the output
        prices_dataframe = self.data.iloc[:, :3]
        returns_dataframe = self.returns.iloc[:, :3]

        # Expected Semi-Covariance matrix
        expected_semi_cov = np.array([[7.302402e-05, 5.855724e-05, 3.075326e-06],
                                      [5.855724e-05, 6.285548e-05, 2.788988e-06],
                                      [3.075326e-06, 2.788988e-06, 3.221170e-06]])

        # Calculating the Semi-Covariance matrix on price data with zero threshold (volatility of negative returns)
        semi_cov = risk_estimators.semi_covariance(prices_dataframe, price_data=True, threshold_return=0)

        # Calculating the Semi-Covariance matrix on returns data with zero threshold (volatility of negative returns)
        semi_cov_ret = risk_estimators.semi_covariance(returns_dataframe, price_data=False, threshold_return=0)

        # Testing if the resulting Semi-Covariance matrix is right
        np.testing.assert_almost_equal(semi_cov, expected_semi_cov, decimal=6)

        # And if the results for price and returns are the same
        np.testing.assert_almost_equal(np.array(semi_cov), np.array(semi_cov_ret), decimal=4)

    def test_exponential_covariance(self):
        """
        Test the calculation of the Exponentially-weighted Covariance matrix.
        """

        risk_estimators = RiskEstimators()

        # Getting first three columns of data to be able to compare the output
        prices_dataframe = self.data.iloc[:, :3]
        returns_dataframe = self.returns.iloc[:, :3]

        # Expected Exponentially-weighted Covariance matrix
        expected_expon_cov = np.array([[2.824303e-04, 3.215506e-04, -4.171518e-06],
                                       [3.215506e-04, 4.585646e-04, -1.868617e-05],
                                       [-4.171518e-06, -1.868617e-05, 8.684991e-06]])

        # Calculating the Exponentially-weighted Covariance matrix on price data with the span of 60
        expon_cov = risk_estimators.exponential_covariance(prices_dataframe, price_data=True, window_span=60)

        # Calculating the Exponentially-weighted Covariance matrix on price data with the span of 60
        expon_cov_ret = risk_estimators.exponential_covariance(returns_dataframe, price_data=False, window_span=60)

        # Testing if the resulting Semi-Covariance matrix is right
        np.testing.assert_almost_equal(expon_cov, expected_expon_cov, decimal=6)

        # And if the results for price and returns are the same
        np.testing.assert_almost_equal(np.array(expon_cov), np.array(expon_cov_ret), decimal=4)
