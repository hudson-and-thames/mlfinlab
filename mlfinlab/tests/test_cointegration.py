"""
Test Cointegration.
"""

import unittest
import os
import pandas as pd

from mlfinlab.statistical_arbitrage import calc_cointegration


class TestStationarity(unittest.TestCase):
    """
    Test Cointegration.
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

    def test_cointegration(self):
        """
        Test Cointegration.
        """
        res = calc_cointegration(self.data.iloc[:, 0], self.data.iloc[:, 1])

        # 6 items in the result.
        self.assertEqual(len(res), 3)

        # Check all values.
        self.assertAlmostEqual(res[0], -1.625729, delta=1e-3)
        self.assertAlmostEqual(res[1], 0.7099084, delta=1e-3)
        self.assertEqual(len(res[2]), 3)

        # Check array list.
        self.assertAlmostEqual(res[2][0], -3.90156503, delta=1e-3)
        self.assertAlmostEqual(res[2][1], -3.3389866, delta=1e-3)
        self.assertAlmostEqual(res[2][2], -3.0464324, delta=1e-3)
