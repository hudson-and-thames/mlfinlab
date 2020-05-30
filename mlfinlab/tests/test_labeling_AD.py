# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.accumulation_distribution import accumulation_distribution


class TestAccumulationDistribution(unittest.TestCase):
    """
    Tests regarding the accumulation/distribution indicator.
    """

    def setUp(self):
        """
        Set the file path for the sample daily high, low, close, volume.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/daily_close.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.idx10 = self.data[:10].index

    def test_base_case(self):
        """
        Test accumulation/distribution with a basic data.
        """
        price = self.data[:10]
        test1 = accumulation_distribution(price)
        test1_actual = pd.Series([53778000, 60768707.6923079, 64156041.97802231, 15423205.05494566, -18183921.15864624,
                                  -10329727.45785893, 3728519.91056171, -46829161.97920199, -11804577.95553338,
                                  -26747139.70049998], index=self.idx10)
        pd.testing.assert_series_equal(test1_actual, test1)

    def test_zeros(self):
        """
        Tests the edge case when the stock does not move up or down at all for the day, i.e. high = low = close. In this
        case, the current money flow volume (CMFV) is assigned to be 0.
        """
        price = self.data[:10]

        # Append a day in which price doesn't change at all
        price = price.append({'Open': 50.0, 'High': 50.0, 'Low': 50.0, 'Close': 50.0, 'Volume': 123456},
                             ignore_index=True)
        test2 = accumulation_distribution(price)
        self.assertTrue(test2.iloc[-1] == test2.iloc[-2])
