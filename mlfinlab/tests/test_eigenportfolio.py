"""
Test Eigenportfolio for Statistical Arbitrage module.
"""

import unittest
import os
import pandas as pd
import numpy as np

from mlfinlab.statistical_arbitrage import calc_pca, calc_all_eigenportfolio


class TestEigenportfolio(unittest.TestCase):
    """
    Test Eigenportfolio.
    """

    def setUp(self):
        """
        Sets the file path for the tick data csv.
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices.csv'
        # Read csv, parse dates, and drop NaN.
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date").dropna(axis=1)

    def test_calc_pca(self):
        """
        Test PCA calculation.
        """
        # Calculate PCA for given data with 1 principal component.
        res = calc_pca(np.array(np.log(self.data)), 1)

        # There are two items in the tuple.
        self.assertEqual(len(res), 2)

        # Length of projection is the same as data.
        self.assertEqual(len(res[0]), 2141)

        # Check some values for projection.
        self.assertAlmostEqual(res[0][5][0], -3.719554, delta=1e-3)
        self.assertAlmostEqual(res[0][100][0], -3.56338343, delta=1e-3)
        self.assertAlmostEqual(res[0][1000][0], 2.6786415, delta=1e-3)

        # Check the length of the second output.
        self.assertEqual(len(res[1]), 23)

        # Check some values for the first principal component.
        self.assertAlmostEqual(res[1][4][0], -0.24159355, delta=1e-3)
        self.assertAlmostEqual(res[1][9][0], -0.233308519, delta=1e-3)
        self.assertAlmostEqual(res[1][22][0], -0.2389528258, delta=1e-3)

    def test_calc_all_eigenportfolio(self):
        """
        Test Eigenportfolio.
        """
        # Calculate PCA for given data with 5 principal components.
        res = calc_all_eigenportfolio(self.data, 5)

        # There are two items in the tuple.
        self.assertEqual(res.shape[0], 8575)
        self.assertEqual(res.shape[1], 23)

        # Check log_ret.
        checking = res.loc['log_ret']
        self.assertEqual(checking.shape[0], 2141)
        self.assertEqual(checking.shape[1], 23)
        self.assertAlmostEqual(checking.iloc[5, 10], 0.0187271, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[286, 20], 0.0018205386, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[1020, -1], 0.00436869, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-7, -5], 0.012938479, delta=1e-3)

        # Check eigenportfolio.
        checking = res.loc['eigenportfolio']
        self.assertEqual(checking.shape[0], 5)
        self.assertEqual(checking.shape[1], 23)
        self.assertAlmostEqual(checking.iloc[2, 2], -0.21104666, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[3, 5], -0.0429993, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-1, -1], -0.19957352, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-3, -5], -0.024271028, delta=1e-3)

        # Check beta.
        checking = res.loc['beta']
        self.assertEqual(checking.shape[0], 6)
        self.assertEqual(checking.shape[1], 23)
        self.assertAlmostEqual(checking.iloc[5, 10], -0.000103154, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[3, 5], -0.000201566, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-1, -1], 0.0001479435, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-3, -5], 0.00122519692, delta=1e-3)

        # Check ret_spread.
        checking = res.loc['ret_spread']
        self.assertEqual(checking.shape[0], 2141)
        self.assertEqual(checking.shape[1], 23)
        self.assertAlmostEqual(checking.iloc[5, 10], 0.0009046264, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[286, 20], 0.0006582330, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[1020, -1], 0.001412272, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-7, -5], -0.0011325141, delta=1e-3)

        # Check cum_resid.
        checking = res.loc['cum_resid']
        self.assertEqual(checking.shape[0], 2141)
        self.assertEqual(checking.shape[1], 23)
        self.assertAlmostEqual(checking.iloc[5, 10], -0.027034045, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[286, 20], 0.0143180950, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[1020, -1], 0.0229146139, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-7, -5], -0.00209802, delta=1e-3)

        # Check z_score.
        checking = res.loc['z_score']
        self.assertEqual(checking.shape[0], 2141)
        self.assertEqual(checking.shape[1], 23)
        self.assertAlmostEqual(checking.iloc[5, 10], 0.780354221, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[286, 20], 0.1498292811, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[1020, -1], 1.01987316572, delta=1e-3)
        self.assertAlmostEqual(checking.iloc[-7, -5], 0.737870045, delta=1e-3)
