"""
Tests the Nested Clustered Optimization (NCO) algorithm.
"""

import unittest
#import os

import numpy as np
import pandas as pd


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


    def test_corr_to_cov(self):
        """
        Test the recovering of the covariance matrix from the correlation matrix.
        """

        nco = NCO()

        # Correlation matrix and the vector of standard deviations
        corr_matrix = np.array([[1, 0.1, -0.1],
                                [0.1, 1, -0.3],
                                [-0.1, -0.3, 1]])
        std_vec = np.array([0.1, 0.2, 0.1])

        # Expected covariance matrix
        expected_matrix = np.array([[ 0.01 ,  0.002, -0.001],
                                    [ 0.002,  0.04 , -0.006],
                                    [-0.001, -0.006,  0.01 ]])

        # Finding the covariance matrix
        cov_matrix = nco.corr_to_cov(corr_matrix, std_vec)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(cov_matrix, expected_matrix, decimal=5)

    def test_cov_to_corr(self):
        """
        Test the deriving of the correlation matrix from a covariance matrix.
        """

        nco = NCO()

        # Covariance matrix
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])

        # Expected correlation matrix
        expected_matrix = np.array([[1, 0.1, -0.1],
                                    [0.1, 1, -0.3],
                                    [-0.1, -0.3, 1]])


        # Finding the covariance matrix
        corr_matrix = nco.cov_to_corr(cov_matrix)

        # Testing the first row of the matrix
        np.testing.assert_almost_equal(corr_matrix, expected_matrix, decimal=5)


    def test_get_pca(self):
        """
        Test the calculation of eigenvalues and eigenvectors from a Hermitian matrix.
        """

        nco = NCO()

        # Correlation matrix as an input
        corr_matrix = np.array([[1, 0.1, -0.1],
                                [0.1, 1, -0.3],
                                [-0.1, -0.3, 1]])

        # Expected correlation matrix
        expected_eigenvalues = np.array([[1.3562, 0,        0],
                                         [0,      0.9438,   0],
                                         [0,      0,      0.7]])
        first_eigenvector = np.array([-3.69048184e-01, -9.29410263e-01,  1.10397126e-16])

        # Finding the eigenvalues
        eigenvalues, eigenvectors = nco.get_pca(corr_matrix)

        # Testing eigenvalues and the first eigenvector
        np.testing.assert_almost_equal(eigenvalues, expected_eigenvalues, decimal=4)
        np.testing.assert_almost_equal(eigenvectors[0], first_eigenvector, decimal=5)

    def test_denoised_corr(self):
        """
        Test the shrinkage the eigenvalues associated with noise.
        """

        nco = NCO()

        # Eigenvalues and eigenvectors to use
        eigenvalues = np.array([[1.3562, 0, 0],
                                [0, 0.9438, 0],
                                [0, 0, 0.7]])
        eigenvectors = np.array([[-3.69048184e-01, -9.29410263e-01,  1.10397126e-16],
                                [-6.57192300e-01,  2.60956474e-01,  7.07106781e-01],
                                [ 6.57192300e-01, -2.60956474e-01,  7.07106781e-01]])

        # Expected correlation matrix
        expected_corr = np.array([[1,  0.13353165, -0.13353165],
                                  [0.13353165,  1, -0.21921986],
                                  [-0.13353165, -0.21921986,  1]])


        # Finding the eigenvalues
        corr_matrix = nco.denoised_corr(eigenvalues, eigenvectors, 1)

        # Testing if the de-noised correlation matrix is right
        np.testing.assert_almost_equal(corr_matrix, expected_corr, decimal=4)

    def de_noised_cov(self):
        """
        Test the shrinkage the eigenvalues associated with noise.
        """

        nco = NCO()

        # Covariance matrix to de-noise and parameters for the theoretical distribution.
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])
        tn_relation = 50
        kde_bwidth = 0.25

        # Expected de-noised covariance matrix
        expected_cov = np.array([[ 0.01,  0.00267029, -0.00133514],
                                 [ 0.00267029,  0.04, -0.00438387],
                                 [-0.00133514, -0.00438387,  0.01]])


        # Finding the de-noised covariance matrix
        cov_matrix = nco.de_noised_cov(cov_matrix, tn_relation, kde_bwidth)

        # Testing if the de-noised covariance matrix is right
        np.testing.assert_almost_equal(cov_matrix, expected_cov, decimal=4)

    def test_cluster_kmeans_base(self):
        """
        Test the finding of the optimal partition of clusters using K-Means algorithm.
        """

        nco = NCO()

        # Correlation matrix and parameters used in the K-Means algorithm.
        np.random.seed(0)
        corr_matrix = pd.DataFrame([[1, 0.1, -0.1],
                                    [0.1, 1, -0.3],
                                    [-0.1, -0.3, 1]])

        max_num_clusters = 2
        n_init = 10

        # Expected correlation matrix of clustered elements, clusters and, Silhouette Coefficient series
        expected_corr = pd.DataFrame([[1,  0.1, -0.1],
                                     [0.1,   1,  -0.3],
                                     [-0.1,- 0.3,   1]])
        expected_clust = {0: [0, 1], 1: [2]}
        expected_silh_coef = pd.Series([0.100834, 0.167626, 0], index=[0, 1, 2])

        # Finding the clustered corresponding values
        corr, clusters, silh_coef = nco.cluster_kmeans_base(corr_matrix, max_num_clusters, n_init)

        # Testing if the values are right
        print(clusters, expected_clust)
        self.assertTrue(clusters == expected_clust)
        np.testing.assert_almost_equal(np.array(corr), np.array(expected_corr), decimal=4)
        np.testing.assert_almost_equal(np.array(silh_coef), np.array(expected_silh_coef), decimal=4)

    def test_opt_port(self):
        """
        Test the estimates of the Convex Optimization Solution (CVO).
        """

        nco = NCO()

        # Covariance matrix
        cov_matrix = np.array([[0.01,   0.002, -0.001],
                               [0.002,   0.04, -0.006],
                               [-0.001, -0.006,  0.01]])

        # Expected weights for minimum variance allocation
        w_expected = np.array([[0.37686939],
                               [0.14257228],
                               [0.48055833]])

        # Finding the optimal weights
        w_cvo = nco.opt_port(cov_matrix, mu_vec=None)

        # Testing if the optimal allocation is right
        np.testing.assert_almost_equal(w_cvo, w_expected, decimal=4)

    def test_opt_port_nco(self):
        """
        Test the estimates the optimal allocation using the (NCO) algorithm
        """

        nco = NCO()

        # Covariance matrix
        cov_matrix = np.array([[0.01,   0.002, -0.001],
                               [0.002,   0.04, -0.006],
                               [-0.001, -0.006,  0.01]])

        # Expected weights for minimum variance allocation
        w_expected = np.array([[0.43875825],
                               [0.09237016],
                               [0.4688716 ]])
        max_num_clusters = 2

        # Finding the optimal weights
        w_nco = nco.opt_port_nco(cov_matrix, max_num_clusters = max_num_clusters)

        # Testing if the optimal allocation is right
        np.testing.assert_almost_equal(w_nco, w_expected, decimal=4)

