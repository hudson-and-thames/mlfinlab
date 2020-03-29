"""
Test various volatility estimates
"""

import unittest
import os
import pandas as pd

from mlfinlab.util import get_parksinson_vol, get_yang_zhang_vol, get_garman_class_vol


class TestVolatilityEstimators(unittest.TestCase):
    """
    Test various volatility estimates (YZ, GS, Parksinson)
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.trades_path = project_path + '/test_data/tick_data.csv'
        self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])
        self.data.index = pd.to_datetime(self.data.index)

    def test_vol_features(self):
        """
        Test volatility estimators
        """
        gm_vol = get_garman_class_vol(self.data.open, self.data.high, self.data.low, self.data.close, window=20)
        yz_vol = get_yang_zhang_vol(self.data.open, self.data.high, self.data.low, self.data.close, window=20)
        park_vol = get_parksinson_vol(self.data.high, self.data.low, window=20)

        self.assertEqual(self.data.shape[0], gm_vol.shape[0])
        self.assertEqual(self.data.shape[0], yz_vol.shape[0])
        self.assertEqual(self.data.shape[0], park_vol.shape[0])

        self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-6)
        self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-6)
        self.assertAlmostEqual(park_vol.mean(), 0.00149997, delta=1e-6)
