# pylint: disable=protected-access
"""
Tests the Nested Clustered Optimization (NCO) algorithm.
"""

import unittest
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
        mu_empir, cov_empir = nco._simulate_covariance(mu_vec, cov_mat, num_obs, False)

        # Testing that with a high number of observations empirical values are close to real ones
        np.testing.assert_almost_equal(mu_empir.flatten(), mu_vec.flatten(), decimal=2)
        np.testing.assert_almost_equal(cov_mat, cov_empir, decimal=2)

        # Also testing the Ledoit-Wolf shrinkage
        mu_empir_shr, cov_empir_shr = nco._simulate_covariance(mu_vec, cov_mat, num_obs, True)
        np.testing.assert_almost_equal(mu_empir_shr.flatten(), mu_vec.flatten(), decimal=2)
        np.testing.assert_almost_equal(cov_mat, cov_empir_shr, decimal=2)

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
        expected_corr = pd.DataFrame([[1, 0.1, -0.1],
                                      [0.1, 1, -0.3],
                                      [-0.1, - 0.3, 1]])
        expected_clust = {0: [0, 1], 1: [2]}
        expected_silh_coef = pd.Series([0.100834, 0.167626, 0], index=[0, 1, 2])

        # Finding the clustered corresponding values
        corr, clusters, silh_coef = nco._cluster_kmeans_base(corr_matrix, max_num_clusters, n_init)

        # When maximum number of clusters is not predefined
        corr_no_max, _, _ = nco._cluster_kmeans_base(corr_matrix, None, n_init)

        # Testing if the values are right
        self.assertTrue(clusters == expected_clust)
        np.testing.assert_almost_equal(np.array(corr), np.array(expected_corr), decimal=4)
        np.testing.assert_almost_equal(np.array(silh_coef), np.array(expected_silh_coef), decimal=4)
        np.testing.assert_almost_equal(np.array(corr), np.array(corr_no_max), decimal=4)

    @staticmethod
    def test_allocate_cvo():
        """
        Test the estimates of the Convex Optimization Solution (CVO).
        """

        nco = NCO()

        # Covariance matrix and the desired mu vector
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])

        mu_vec = np.array([1, 1, 1]).reshape(-1, 1)

        # Expected weights for minimum variance allocation
        w_expected = np.array([[0.37686939],
                               [0.14257228],
                               [0.48055833]])

        # Finding the optimal weights
        w_cvo = nco.allocate_cvo(cov_matrix, mu_vec=None)

        # Also when manually inputting the vector mu
        w_cvo_mu = nco.allocate_cvo(cov_matrix, mu_vec=mu_vec)

        # Testing if the optimal allocation is right and if the custom mu works
        np.testing.assert_almost_equal(w_cvo, w_expected, decimal=4)
        np.testing.assert_almost_equal(w_cvo, w_cvo_mu, decimal=4)

    @staticmethod
    def test_allocate_nco():
        """
        Test the estimates the optimal allocation using the (NCO) algorithm
        """

        nco = NCO()

        # Covariance matrix and the custom mu vector
        np.random.seed(0)
        cov_matrix = np.array([[0.01, 0.002, -0.001],
                               [0.002, 0.04, -0.006],
                               [-0.001, -0.006, 0.01]])

        mu_vec = np.array([1, 1, 1]).reshape(-1, 1)

        # Expected weights for minimum variance allocation
        w_expected = np.array([[0.43875825],
                               [0.09237016],
                               [0.4688716]])
        max_num_clusters = 2

        # Finding the optimal weights
        w_nco = nco.allocate_nco(cov_matrix, max_num_clusters=max_num_clusters)

        # Finding the optimal weights using the custom mu vector
        w_nco_mu = nco.allocate_nco(cov_matrix, mu_vec=mu_vec, max_num_clusters=max_num_clusters)

        # Testing if the optimal allocation is right and if the custom mu works
        np.testing.assert_almost_equal(w_nco, w_expected, decimal=4)
        np.testing.assert_almost_equal(w_nco, w_nco_mu, decimal=4)

    @staticmethod
    def test_allocate_mcos():
        """
        Test the estimates of the optimal allocation using the Monte Carlo optimization selection
        """

        nco = NCO()

        # Covariance matrix, mean vector and other variables for the method
        np.random.seed(0)
        mu_vec = np.array([0, 0.1, 0.2, 0.3])
        cov_mat = np.array([[1, 0.1, 0.2, 0.3],
                            [0.1, 1, 0.1, 0.2],
                            [0.2, 0.1, 1, 0.1],
                            [0.3, 0.2, 0.1, 1]])
        num_obs = 100
        num_sims = 2
        kde_bwidth = 0.25
        min_var_portf = True
        lw_shrinkage = False

        # Alternative set of values
        min_var_portf_alt = False
        kde_bwidth_alt = 0

        # Expected weights for minimum variance allocation
        w_cvo_expected = pd.DataFrame([[0.249287, 0.256002, 0.242593, 0.252118],
                                       [0.257547, 0.265450, 0.242453, 0.234551]])

        w_nco_expected = pd.DataFrame([[0.248396, 0.243172, 0.250751, 0.257680],
                                       [0.257547, 0.265450, 0.242453, 0.234551]])

        # Expected weights for maximum Sharpe ratio allocation
        w_cvo_sr_expected = pd.DataFrame([[-1.081719, 1.810936, 1.218067, 3.978880],
                                          [-2.431651, 0.594868, -0.210175, 5.117628]])

        w_nco_sr_expected = pd.DataFrame([[-1.060835, 1.910503, 1.315026, 3.908128],
                                          [-0.937168, 1.886158, -0.389275, 4.884809]])

        # Finding the optimal weights for minimum variance
        w_cvo, w_nco = nco.allocate_mcos(mu_vec, cov_mat, num_obs, num_sims, kde_bwidth, min_var_portf, lw_shrinkage)

        # Finding the optimal weights for maximum Sharpe ratio
        w_cvo_sr, w_nco_sr = nco.allocate_mcos(mu_vec, cov_mat, num_obs, num_sims, kde_bwidth_alt, min_var_portf_alt, lw_shrinkage)

        # Testing if the optimal allocation simulations are right
        np.testing.assert_almost_equal(np.array(w_cvo), np.array(w_cvo_expected), decimal=4)
        np.testing.assert_almost_equal(np.array(w_nco), np.array(w_nco_expected), decimal=4)

        np.testing.assert_almost_equal(np.array(w_cvo_sr), np.array(w_cvo_sr_expected), decimal=4)
        np.testing.assert_almost_equal(np.array(w_nco_sr), np.array(w_nco_sr_expected), decimal=4)

    @staticmethod
    def test_estim_errors_mcos():
        """
        Test the computation the true optimal allocation w, and compares that result with the estimated ones by MCOS.
        """

        nco = NCO()

        # Weights from CVO and NCO, and data to estimate the true optimal weights
        np.random.seed(0)
        w_cvo = pd.DataFrame([[0.249287, 0.256002, 0.242593, 0.252118],
                              [0.257547, 0.265450, 0.242453, 0.234551]])

        w_nco = pd.DataFrame([[0.248396, 0.243172, 0.250751, 0.257680],
                              [0.257547, 0.265450, 0.242453, 0.234551]])

        mu_vec = np.array([0, 0.1, 0.2, 0.3])
        cov_mat = np.array([[1, 0.1, 0.2, 0.3],
                            [0.1, 1, 0.1, 0.2],
                            [0.2, 0.1, 1, 0.1],
                            [0.3, 0.2, 0.1, 1]])
        min_var_portf = True

        # Expected errors
        err_cvo_expected = 0.0062604
        err_nco_expected = 0.0111115

        # Finding the errors in estimations
        err_cvo, err_nco = nco.estim_errors_mcos(w_cvo, w_nco, mu_vec, cov_mat, min_var_portf)

        # Testing if the error computation is right
        np.testing.assert_almost_equal(err_cvo, err_cvo_expected, decimal=4)
        np.testing.assert_almost_equal(err_nco, err_nco_expected, decimal=4)

    @staticmethod
    def test_form_block_matrix():
        """
        Test the creation of a block correlation matrix with given parameters.
        """

        nco = NCO()

        # Parameters to build the block correlation matrix with
        num_blocks = 2
        block_size = 2
        block_corr = 0.3

        # Expected corr matrix
        corr_expected = np.array([[1, 0.3, 0, 0],
                                  [0.3, 1, 0, 0],
                                  [0, 0, 1, 0.3],
                                  [0, 0, 0.3, 1]])

        # Finding the errors in estimations
        corr_matrix = nco._form_block_matrix(num_blocks, block_size, block_corr)

        # Testing if the error computation is right
        np.testing.assert_almost_equal(corr_matrix, corr_expected, decimal=4)

    @staticmethod
    def test_form_true_matrix():
        """
        Test the creation of a random vector of means and a random covariance matrix.
        """

        nco = NCO()

        # Parameters to build the vector of means and a covariance matrix
        np.random.seed(0)
        num_blocks = 2
        block_size = 2
        block_corr = 0.3
        std = 0.3

        # Alternative std parameter
        std_alt = None

        # Expected vector of means and covariance matrix
        mu_expected = np.array([[0.5936214],
                                [0.97226796],
                                [0.8602674],
                                [0.00681664]])

        cov_expected = pd.DataFrame([[0.09, 0.027, 0, 0],
                                     [0.027, 0.09, 0, 0],
                                     [0, 0, 0.09, 0.027],
                                     [0, 0, 0.027, 0.09]])

        mu_alt_expected = np.array([[0.34260886],
                                    [0.12059827],
                                    [0.24366456],
                                    [0.17248975]])

        # Finding the random vector of means and covariance matrix
        mu_vec, cov_matrix = nco.form_true_matrix(num_blocks, block_size, block_corr, std)

        # Also when the std is default
        mu_vec_alt, _ = nco.form_true_matrix(num_blocks, block_size, block_corr, std_alt)

        # Testing if the results are right
        np.testing.assert_almost_equal(mu_vec, mu_expected, decimal=4)
        np.testing.assert_almost_equal(np.array(cov_matrix), np.array(cov_expected), decimal=4)

        np.testing.assert_almost_equal(mu_vec_alt, mu_alt_expected, decimal=4)
