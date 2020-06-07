"""
Test Stationarity
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.statistical_arbitrage import calc_stationarity


class TesStationarity(unittest.TestCase):
    """
    Test Stationarity.
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

    def test_stationarity(self):
        """
        Test stationarity.
        """
        res = calc_stationarity(self.data.iloc[:, 0])

        # 6 items in the result.
        self.assertEqual(len(res), 6)

        self.assertAlmostEqual(res[0], -2.37269, delta=1e-3)
        self.assertAlmostEqual(res[1], 0.149591, delta=1e-3)
        self.assertEqual(res[2], 20)
        self.assertEqual(res[3], 2120)
        self.assertEqual(len(res[4]), 3)
        self.assertAlmostEqual(res[4].get('1%'), -3.433438, delta=1e-3)
        self.assertAlmostEqual(res[4].get('5%'), -2.86290, delta=1e-3)
        self.assertAlmostEqual(res[4].get('10%'), -2.567496, delta=1e-3)
        self.assertAlmostEqual(res[5], 4514.87921, delta=1e-3)
