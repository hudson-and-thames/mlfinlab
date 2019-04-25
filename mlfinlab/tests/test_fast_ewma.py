"""
Tests ewma function from fast_ewma module
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.util.fast_ewma import ewma


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Imbalance Dollar bars
    2. Imbalance Volume bars
    3. Imbalance Tick bars
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/tick_data.csv'

    def test_ewma(self):
        """
        Tests the imbalance dollar bars implementation.
        """
        test_sample = pd.read_csv(self.path)
        price_arr = np.array(test_sample.Price.values, dtype=float)
        ewma_res = ewma(price_arr, window=20)

        # Assert output array length equals input array length
        self.assertTrue(ewma_res.shape == price_arr.shape)
        # Assert the first value of ewma equals to input array value
        self.assertTrue(ewma_res[0] == price_arr[0])
        # Assert next value check with tolerance of 1e-5
        self.assertTrue(abs(ewma_res[1] - 1100.00) < 1e-5)
