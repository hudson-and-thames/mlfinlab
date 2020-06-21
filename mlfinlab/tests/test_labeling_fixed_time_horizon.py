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
        self.idx10 = self.data[:10].index

    def test_basic(self):
        """
        Tests for basic cases, constant threshold and no standardization, varied lookforward periods
        """
        close = self.data['SPY'][:10]
        test1 = fixed_time_horizon(close, 0, look_forward=1)
        test2 = fixed_time_horizon(close, 0, look_forward=3)
        test3 = fixed_time_horizon(close, 0.01, look_forward=1)
        test4 = fixed_time_horizon(close, 1.0, look_forward=2)
        test1_actual = pd.Series([-1, -1, -1, -1, 1, 1, -1, 1, -1, np.nan], index=self.idx10)
        test2_actual = pd.Series([-1, -1, -1, 1, 1, 1, -1, np.nan, np.nan, np.nan], index=self.idx10)
        test3_actual = pd.Series([0, -1, 0, -1, 1, 0, 0, 0, -1, np.nan], index=self.idx10)
        test4_actual = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan], index=self.idx10)
        pd.testing.assert_series_equal(test1_actual, test1)
        pd.testing.assert_series_equal(test2_actual, test2)
        pd.testing.assert_series_equal(test3_actual, test3)
        pd.testing.assert_series_equal(test4_actual, test4)

    def test_dynamic_threshold(self):
        """
        Tests for when threshold is a pd.Series rather than a constant
        """
        close = self.data['SPY'][:10]
        threshold1 = pd.Series([0.01, 0.005, 0, 0.01, 0.02, 0.03, 0.1, -1, 0.99, 0], index=self.data[:10].index)

        test5 = fixed_time_horizon(close, threshold1, look_forward=1)
        test6 = fixed_time_horizon(close, threshold1, look_forward=4)
        test5_actual = pd.Series([0, -1, -1, -1, 0, 0, 0, 1, 0, np.nan], index=self.idx10)
        test6_actual = pd.Series([-1, -1, -1, 0, 0, 0, np.nan, np.nan, np.nan, np.nan], index=self.idx10)
        pd.testing.assert_series_equal(test5_actual, test5)
        pd.testing.assert_series_equal(test6_actual, test6)

    def test_with_standardization(self):
        """
        Test cases with standardization, with constant and dynamic threshold
        """
        close = self.data['SPY'][:10]
        threshold2 = pd.Series([1, 2, 0, 0.5, 0.02, 1.5, 10, -1, 500, 1], index=self.data[:10].index)

        test7 = fixed_time_horizon(close, 1, look_forward=1, standardized=True, window=4)
        test8 = fixed_time_horizon(close, 0.1, look_forward=1, standardized=True, window=5)
        test9 = fixed_time_horizon(close, threshold2, look_forward=2, standardized=True, window=3)
        test7_actual = pd.Series([np.nan, np.nan, np.nan, 0, 1, 0, 0, 0, -1, np.nan], index=self.idx10)
        test8_actual = pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, -1, 1, -1, np.nan], index=self.idx10)
        test9_actual = pd.Series([np.nan, np.nan, 1, 1, 1, 0, 0, -1, np.nan, np.nan], index=self.idx10)
        pd.testing.assert_series_equal(test7_actual, test7)
        pd.testing.assert_series_equal(test8_actual, test8)
        pd.testing.assert_series_equal(test9_actual, test9)

    def test_look_forward_warning(self):
        """
        Verifies that the correct warning is raised if look_forward is greater than the length of the data
        """
        close = self.data['SPY'][:10]
        with self.assertWarns(UserWarning):
            labels = fixed_time_horizon(close, 0.01, look_forward=50)
        np.testing.assert_allclose(labels, [np.nan]*len(self.data['SPY'][:10]))

    def test_standardization_warning(self):
        """
        Verify that an exception is raised if standardization is set to true, but window is not specified as an int.
        Checks warning when look_forward is greater than the length of the data series
        """
        close = self.data['SPY'][:10]
        with self.assertRaises(Exception):
            fixed_time_horizon(close, 0.01, look_forward=50, standardized=True)
        with self.assertWarns(UserWarning):
            labels = fixed_time_horizon(close, 0.01, look_forward=50, standardized=True, window=50)
        np.testing.assert_allclose(labels, [np.nan]*len(self.data['SPY'][:10]))
