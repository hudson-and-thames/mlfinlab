"""
Test Cointegration.
"""

import unittest
import os
import pandas as pd
import numpy as np

from mlfinlab.statistical_arbitrage import calc_engle_granger, calc_johansen


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

    def test_engle_granger(self):
        """
        Test Cointegration with Engle-Granger.
        """
        res = calc_engle_granger(self.data.iloc[:, 0], self.data.iloc[:, 1])

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

    def test_johansen(self):
        """
        Test Cointegration with Johansen.
        """
        res = calc_johansen(self.data, 0, 1)

        # CVM.
        self.assertTrue(np.isnan(res[0][5][1]))
        self.assertAlmostEqual(res[0][14][2], 64.996, delta=1e-3)

        # CVT.
        self.assertTrue(np.isnan(res[1][3][2]))
        self.assertAlmostEqual(res[1][15][1], 159.529, delta=1e-3)

        # EIG.
        self.assertAlmostEqual(res[2][3], 0.05269557870803097, delta=1e-3)

        # EVEC.
        self.assertAlmostEqual(res[3][12][2], 0.2027615, delta=1e-3)

        # IND.
        self.assertEqual(res[4].tolist(), np.arange(23).tolist())

        # LR1.
        self.assertAlmostEqual(res[5][5], 659.417581, delta=1e-3)

        # LR2.
        self.assertAlmostEqual(res[6][12], 35.344217, delta=1e-3)

        # MAX_EIG_STAT.
        self.assertAlmostEqual(res[7][8], 66.640731, delta=1e-3)

        # MAX_EIG_STAT_CRIT_VALS.
        self.assertTrue(np.isnan(res[8][2][1]))
        self.assertAlmostEqual(res[8][18][1], 33.8777, delta=1e-3)

        # METH.
        self.assertEqual(res[9], 'johansen')

        # R0T.
        self.assertAlmostEqual(res[10][567][3], 0.022608, delta=1e-3)

        # RKT.
        self.assertAlmostEqual(res[11][2][10], 7.4506079, delta=1e-3)

        # TRACE_STAT.
        self.assertAlmostEqual(res[12][0], 1365.41034, delta=1e-3)

        # TRACE_STAT_CRIT_VALS.
        self.assertTrue(np.isnan(res[13][5][1]))
        self.assertAlmostEqual(res[13][20][0], 27.0669, delta=1e-3)
