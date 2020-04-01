# pylint: disable=not-callable, unused-import, invalid-name, inconsistent-return-statements
"""
Tests the Nested Clustered Optimization (RiskEstimators) algorithm.
"""

import unittest
import types
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


class TestRiskEstimators(unittest.TestCase):
    """
    Tests different functions of the RiskEstimators class.
    """

    def setUp(self):
        """
        Initialize
        """
        self.codeAttribute = '__code__' if sys.version_info[0] == 3 else 'func_code'

    @staticmethod
    def free_var(val):
        """
        Allows to pass variables to nested functions

        :param val: (any) Value of a parameter
        :return: (cell) Closure of the nested function
        """

        def nested():
            return val

        return nested.__closure__[0]

    def nested(self, outer, inner_name, **freeVars):
        """
        Allows to access the nested functions for unit tests.

        :param outer: (function) The parent function
        :param innerName: (str) The name of the nested function
        :param freeVars: (any) Variables to pass to a parent and nested function
        :return: (function) The nested function
        """

        if isinstance(outer, (types.FunctionType, types.MethodType)):
            outer = outer.__getattribute__(self.codeAttribute)

        for const in outer.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == inner_name:

                return types.FunctionType(const, globals(), None, None, tuple(
                    self.free_var(freeVars[name]) for name in const.co_freevars))

    def test_mp_pdf(self):
        """
        Test the deriving of pdf of the Marcenko-Pastur distribution.
        """

        risk_estimators = RiskEstimators()

        nested_mp_pdf = self.nested(risk_estimators.de_noised_cov, 'mp_pdf')

        # Properties for the distribution
        var = 0.1
        tn_relation = 5
        num_points = 5

        # Calculating the pdf in 5 points
        pdf_mp = nested_mp_pdf(var, tn_relation, num_points)

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

        nested_fit_kde = self.nested(risk_estimators.de_noised_cov, 'fit_kde')

        # Values to fit to and parameters
        observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        eval_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Calculating the pdf in 7 chosen points
        pdf_kde = nested_fit_kde(observations, 0.25, 'gaussian', eval_points)

        # Testing the values and if the pdf is symmetric
        self.assertEqual(pdf_kde[0.0], pdf_kde[0.6])
        self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
        self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
        self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-5)

        # Testing also on unique values of the set as a default output
        pdf_kde_default = nested_fit_kde(observations, 0.25, 'gaussian', eval_points)
        self.assertEqual(pdf_kde[0.1], pdf_kde_default[0.1])
        self.assertEqual(pdf_kde_default[0.2], pdf_kde_default[0.4])

    def test_corr_to_cov(self):
        """
        Test the recovering of the covariance matrix from the correlation matrix.
        """

        risk_estimators = RiskEstimators()

        nested_corr_to_cov = self.nested(risk_estimators.de_noised_cov, 'corr_to_cov')

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
        cov_matrix = nested_corr_to_cov(corr_matrix, std_vec)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(cov_matrix, expected_matrix, decimal=5)

    def test_cov_to_corr(self):
        """
        Test the deriving of the correlation matrix from a covariance matrix.
        """

        risk_estimators = RiskEstimators()

        nested_cov_to_corr = self.nested(risk_estimators.de_noised_cov, 'cov_to_corr')

        # Covariance matrix
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])

        # Expected correlation matrix
        expected_matrix = np.array([[1, 0.1, -0.1],
                                    [0.1, 1, -0.3],
                                    [-0.1, -0.3, 1]])


        # Finding the covariance matrix
        corr_matrix = nested_cov_to_corr(cov_matrix)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(corr_matrix, expected_matrix, decimal=5)

    def test_get_pca(self):
        """
        Test the calculation of eigenvalues and eigenvectors from a Hermitian matrix.
        """

        risk_estimators = RiskEstimators()

        nested_get_pca = self.nested(risk_estimators.de_noised_cov, 'get_pca')

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
        eigenvalues, eigenvectors = nested_get_pca(corr_matrix)

        # Testing eigenvalues and the first eigenvector
        np.testing.assert_almost_equal(eigenvalues, expected_eigenvalues, decimal=4)
        np.testing.assert_almost_equal(eigenvectors[0], first_eigenvector, decimal=5)

    @staticmethod
    def test_de_noised_cov():
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
        cov_matrix = risk_estimators.de_noised_cov(cov_matrix, tn_relation, kde_bwidth)

        # Testing if the de-noised covariance matrix is right
        np.testing.assert_almost_equal(cov_matrix, expected_cov, decimal=4)
