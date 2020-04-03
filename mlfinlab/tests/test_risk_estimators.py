# pylint: disable=protected-access
"""
Tests the functions from the RiskEstimators class.
"""

import unittest
import numpy as np
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


class TestRiskEstimators(unittest.TestCase):
    """
    Tests different functions of the RiskEstimators class.
    """

    def setUp(self):
        """
        Initialize
        """

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
        pdf_kde = risk_estimators._fit_kde(observations, eval_points=eval_points)

        # Testing the values and if the pdf is symmetric
        self.assertEqual(pdf_kde[0.0], pdf_kde[0.6])
        self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
        self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
        self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-5)

        # Testing also on unique values of the set as a default output
        pdf_kde_default = risk_estimators._fit_kde(observations)
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
