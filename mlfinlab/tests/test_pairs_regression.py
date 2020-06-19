"""
Test Pairs Regression.
"""

import unittest
import os
import pandas as pd
import numpy as np

from mlfinlab.statistical_arbitrage import calc_all_regression, calc_rolling_regression


class TestPairsRegression(unittest.TestCase):
    """
    Test Pairs Regression.
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

    def test_all_regression(self):
        """
        Test all regression.
        """
        res = calc_all_regression(self.data.iloc[:, 0], self.data.iloc[:, 1])

        # Check shape.
        self.assertEqual(res.shape[0], 2141)
        self.assertEqual(res.shape[1], 9)

        # Check all values.
        self.assertAlmostEqual(res.iloc[:, 0][3], 48.57666, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 1][279], 15.899999, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 2][1027], -0.01450508, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 3][586], -0.00373308, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 4][1788], 0.75397453, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 5][2000], 0, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 6][786], 0.00270831, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 7][182], -0.0668679, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 8][999], -1.4291310, delta=1e-3)

    def test_rolling_regression(self):
        """
        Test rolling regression.
        """

        # Singular matrix problem within window of 2.
        res = calc_rolling_regression(self.data.iloc[:, 3], self.data.iloc[:, 4], window=2)

        # Calculated results.
        res = calc_rolling_regression(self.data.iloc[:, 3], self.data.iloc[:, 4], window=5)

        # Check shape.
        self.assertEqual(res.shape[0], 2141)
        self.assertEqual(res.shape[1], 6)

        # Check all values.
        self.assertTrue(np.isnan(res.iloc[:, 0][3]))
        self.assertAlmostEqual(res.iloc[:, 1][279], 38.669998, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 2][1027], 1.128547746, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 3][586], -53.5777879, delta=1e-3)
        self.assertAlmostEqual(res.iloc[:, 4][1788], -0.096600805, delta=1e-3)
