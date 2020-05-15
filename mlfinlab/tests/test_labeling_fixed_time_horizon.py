# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.fixed_time_horizon import fixed_time_horizon


class TestLabellingFixedTime(unittest.TestCase):
    """
    Tests regarding fixed time horizon labelling method
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.data.index = pd.to_datetime(self.data.index)

    def test_fixed_time_horizon(self):
        """
        Assert that the fixed time horizon labelling works as expected.
        Checks a range of static and dynamic thresholds.
        """
        close = self.data['SPY'][:10]
        test_threshold1 = pd.Series([0.01, 0.005, 0, 0.01, 0.02, 0.03, 0.1, -1, 0.99, 0], index=close.index)
        test_threshold2 = pd.Series([1, 2, 0, 0.5, 0.02, 1.5, 10, -1, 500, 1], index=close.index)
        test_standardize = [(0.1, 0.1), (-0.025, 0.01), (0, 0.01), (0, 0.01), (0, 0.01), (0.1, 0.1), (-0.025, 0.01),
                            (0, 0.9), (-9999, 0.01), (0, 0.01)]

        # Test cases without standardization
        test1 = fixed_time_horizon(close, 0, 1)
        test2 = fixed_time_horizon(close, 0, 3)
        test3 = fixed_time_horizon(close, 0.01, 1)
        test4 = fixed_time_horizon(close, test_threshold1, 1)
        test5 = fixed_time_horizon(close, 0.05, 3)
        test1_actual = np.array([-1, -1, -1, -1, 1, 1, -1, 1, -1, np.nan])
        test2_actual = np.array([-1, -1, -1, 1, 1, 1, -1, np.nan, np.nan, np.nan])
        test3_actual = np.array([0, -1, 0, -1, 1, 0, 0, 0, -1, np.nan])
        test4_actual = np.array([0, -1, -1, -1, 0, 0, 0, 1, 0., np.nan])
        test5_actual = np.array([0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan])
        np.testing.assert_allclose(test1_actual, test1)
        np.testing.assert_allclose(test2_actual, test2)
        np.testing.assert_allclose(test3_actual, test3)
        np.testing.assert_allclose(test4_actual, test4)
        np.testing.assert_allclose(test5_actual, test5)

        # Test with standardization
        test6 = fixed_time_horizon(close, 1, lookfwd=1, standardized=test_standardize)
        test7 = fixed_time_horizon(close, test_threshold2, lookfwd=2, standardized=test_standardize)
        test6_actual = np.array([-1., 0., 0., -1., 1., 0., 1., 0., 1., np.nan])
        test7_actual = np.array([-1., 0., -1., -1., 1., 0., 0., 1., np.nan, np.nan])
        np.testing.assert_allclose(test6_actual, test6)
        np.testing.assert_allclose(test7_actual, test7)
