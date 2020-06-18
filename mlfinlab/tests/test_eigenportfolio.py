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
        res = calc_pca(np.log(self.data), 1)

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

    def test_calc_all_pca(self):
        """
        Test Eigenportfolio and PCA calculation.
        """
        # Calculate PCA for given data with 1 principal component.
        res = calc_all_eigenportfolio(np.log(self.data), 1)

        # There are two items in the tuple.
        self.assertEqual(res.shape[0], 4284)
        self.assertEqual(res.shape[1], 23)

        # Length of projection is the same as data.
        self.assertEqual(res.index[0][0], 'spread')
        self.assertEqual(res.index[-1][1], 'constants')

        # Check the values for spreads.
        self.assertAlmostEqual(res.iloc[5, 10], 0.186152852268, delta=1e-3)
        self.assertAlmostEqual(res.iloc[37, 5], -0.146775127, delta=1e-3)
        self.assertAlmostEqual(res.iloc[586, 0], 0.135358564, delta=1e-3)
        self.assertAlmostEqual(res.iloc[1023, 10], -0.163594463, delta=1e-3)
        self.assertAlmostEqual(res.iloc[-5, -3], 0.03036983904, delta=1e-3)

        # There are two terms: Eigenportfolio 0 and Constants.
        self.assertEqual(len(res.loc['eigenportfolio']), 2)

        # Check some values for the first principal component.
        self.assertAlmostEqual(res.loc['eigenportfolio'].iloc[0, 6], -0.0301915332, delta=1e-3)
        self.assertAlmostEqual(res.loc['eigenportfolio'].iloc[1, 3], 3.75083698, delta=1e-3)
        self.assertAlmostEqual(res.loc['eigenportfolio'].iloc[1, 10], 2.6658233048, delta=1e-3)
        self.assertAlmostEqual(res.loc['eigenportfolio'].iloc[0, 20], -0.00400423, delta=1e-3)
        self.assertAlmostEqual(res.loc['eigenportfolio'].iloc[-1, -1], 4.86810379, delta=1e-3)
