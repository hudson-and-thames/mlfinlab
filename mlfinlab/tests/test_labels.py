"""
Test various filters.
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.labeling.labeling import get_daily_vol


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Dollar bars
    2. Volume bars
    3. Tick bars
    """
    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

    def test_daily_volatility(self):
        """
        Daily vol as implemented here matches the code in the book.
        Although I have reservations, example: no minimum value is set in the EWM.
        Thus it returns values for volatility before there are even enough data points.
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)

        # Last value in the set is still the same
        self.assertTrue(daily_vol[-1] == 0.008968238932170641)

        # Size matches
        self.assertTrue(daily_vol.shape[0] == 960)
