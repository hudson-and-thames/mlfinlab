"""
Tests the Nested Clustered Optimization (NCO) algorithm.
"""

import unittest
#import os

import numpy as np
#import pandas as pd


from mlfinlab.portfolio_optimization.nco import NCO


class TestNCO(unittest.TestCase):
    """
    Tests different functions of the NCO algorithm class.
    """

    def setUp(self):
        """
        Initialize
        """

    @staticmethod
    def test_simulate_covariance():
        """
        Test the deriving an empirical vector of means and an empirical covariance matrix.
        """

        nco = NCO()

        # Real values used for deriving empirical ones
        mu_vec = np.array([0, 0.1, 0.2, 0.3])
        cov_mat = np.array([[1, 0.1, 0.2, 0.3],
                            [0.1, 1, 0.1, 0.2],
                            [0.2, 0.1, 1, 0.1],
                            [0.3, 0.2, 0.1, 1]])
        num_obs = 100000

        # Using the function
        mu_empir, cov_empir = nco.simulate_covariance(mu_vec, cov_mat, num_obs, False)

        # Testing that with a high number of observations empirical values are close to real ones
        np.testing.assert_almost_equal(mu_empir.flatten(), mu_vec.flatten(), decimal=2)
        np.testing.assert_almost_equal(cov_mat, cov_empir, decimal=2)

        # Also testing the Ledoit-Wolf shrinkage
        mu_empir_shr, cov_empir_shr = nco.simulate_covariance(mu_vec, cov_mat, num_obs, True)
        np.testing.assert_almost_equal(mu_empir_shr.flatten(), mu_vec.flatten(), decimal=2)
        np.testing.assert_almost_equal(cov_mat, cov_empir_shr, decimal=2)

    def test_mp_pdf(self):
        """
        Test the deriving of pdf of the Marcenko-Pastur distribution.
        """

        nco = NCO()

        # Properties for the distribution
        var = 0.1
        tn_relation = 5
        num_points = 5

        # Calculating the pdf in 5 points
        pdf_mp = nco.mp_pdf(var, tn_relation, num_points)

        # Testing the minimum and maximum and non-zero values of the pdf
        self.assertAlmostEqual(pdf_mp.index[0], 0.03056, delta=1e-4)
        self.assertAlmostEqual(pdf_mp.index[4], 0.20944, delta=1e-4)

        # Testing that the distribution curve is right
        self.assertTrue(pdf_mp.values[1] > pdf_mp.values[2] > pdf_mp.values[3])


    def test_fit_kde(self):
        """
        Test the kernel fitting to a series of observations.
        """

        nco = NCO()

        # Values to fit to and parameters
        observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        eval_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Calculating the pdf in 7 chosen points
        pdf_kde = nco.fit_kde(observations, eval_points=eval_points)

        # Testing the values and if the pdf is symmetric
        self.assertEqual(pdf_kde[0.0], pdf_kde[0.6])
        self.assertEqual(pdf_kde[0.1], pdf_kde[0.5])
        self.assertEqual(pdf_kde[0.2], pdf_kde[0.4])
        self.assertAlmostEqual(pdf_kde[0.3], 1.44413, delta=1e-5)

        # Testing also on unique values of the set as a default output
        pdf_kde_default = nco.fit_kde(observations)
        self.assertEqual(pdf_kde[0.1], pdf_kde_default[0.1])
        self.assertEqual(pdf_kde_default[0.2], pdf_kde_default[0.4])

    def test_pdf_fit(self):
        """
        Test the fit between empirical pdf and the theoretical pdf.
        """

        nco = NCO()

        # Values to calculate theoretical and empirical pdfs
        var = 0.6
        eigen_observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        tn_relation = 2
        kde_bwidth = 0.4

        # Calculating the SSE
        pdf_kde = nco.pdf_fit(var, eigen_observations, tn_relation, kde_bwidth)

        # Testing the SSE value
        self.assertAlmostEqual(pdf_kde, 50.51326, delta=1e-5)


    def test_find_max_eval(self):
        """
        Test the search for maximum random eigenvalue.
        """

        nco = NCO()

        # Values to calculate theoretical and empirical pdfs
        eigen_observations = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5])
        tn_relation = 2
        kde_bwidth = 0.4

        # Optimizing and getting the maximum random eigenvalue and the optimal variation
        maximum_eigen, var = nco.find_max_eval(eigen_observations, tn_relation, kde_bwidth)

        # Testing the maximum random eigenvalue and the optimal variation
        self.assertAlmostEqual(maximum_eigen, 2.41011, delta=1e-5)
        self.assertAlmostEqual(var, 0.82702, delta=1e-5)
