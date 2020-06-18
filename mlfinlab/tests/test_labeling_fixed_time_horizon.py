# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.fixed_time_horizon import fixed_time_horizon


class TestLabelingFixedTime(unittest.TestCase):
    """
    Tests regarding fixed time horizon labeling method.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.data.index = pd.to_datetime(self.data.index)
        self.idx10 = self.data[:10].index

    def test_basic(self):
        """
        Tests for basic case, constant threshold and no standardization, lag.
        """
        close = self.data[['SPY', 'EPP', 'FXI']][:10]
        test1 = fixed_time_horizon(close['SPY'], lag=False)
        test2 = fixed_time_horizon(close, lag=False)
        test3 = fixed_time_horizon(close, lag=True)
        test4 = fixed_time_horizon(close, threshold=0.01, lag=True)
        test5 = fixed_time_horizon(close['SPY'], threshold=0.99, lag=False)
        test1_actual = pd.Series([np.nan, -1, -1, -1, -1, 1, 1, -1, 1, -1], index=self.idx10)
        test2_actual = pd.DataFrame({'SPY': [np.nan, -1, -1, -1, -1, 1, 1, -1, 1, -1],
                                     'EPP': [np.nan, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                                     'FXI': [np.nan, -1, -1, 1, -1, 1, 1, -1, 1, -1]}, index=self.idx10)
        test3_actual = pd.DataFrame({'SPY': [-1, -1, -1, -1, 1, 1, -1, 1, -1, np.nan],
                                     'EPP': [1, -1, 1, -1, 1, 1, -1, 1, -1, np.nan],
                                     'FXI': [-1, -1, 1, -1, 1, 1, -1, 1, -1, np.nan]}, index=self.idx10)
        test4_actual = pd.DataFrame({'SPY': [0, -1, 0, -1, 1, 0, 0, 0, -1, np.nan],
                                     'EPP': [0, -1, 1, -1, 1, 1, -1, 1, -1, np.nan],
                                     'FXI': [0, -1, 1, -1, 1, 0, -1, 0, -1, np.nan]}, index=self.idx10)
        test5_actual = pd.Series([np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0], index=self.idx10)

        pd.testing.assert_series_equal(test1_actual, test1, check_names=False)
        pd.testing.assert_frame_equal(test2_actual, test2)
        pd.testing.assert_frame_equal(test3_actual, test3)
        pd.testing.assert_frame_equal(test4_actual, test4)
        pd.testing.assert_series_equal(test5_actual, test5, check_names=False)

    def test_dynamic_threshold(self):
        """
        Tests for when threshold is a pd.Series rather than a constant.
        """
        close = self.data[['SPY', 'EPP', 'FXI']][:10]
        threshold1 = pd.Series([0.01, 0.005, 0, 0.01, 0.02, 0.03, 0.1, -1, 0.99, 0], index=self.idx10)
        test6 = fixed_time_horizon(close, threshold=threshold1, lag=True)
        test7 = fixed_time_horizon(close['SPY'], threshold=threshold1, lag=False)
        test6_actual = pd.DataFrame({'SPY': [0, -1, -1, -1, 0, 0, 0, 1, 0, np.nan],
                                     'EPP': [0, -1, 1, -1, 0, 0, 0, 1, 0, np.nan],
                                     'FXI': [0, -1, 1, -1, 1, 0, 0, 1, 0, np.nan]}, index=self.idx10)
        test7_actual = pd.Series([np.nan, 0, -1, 0, 0, 0, 0, 1, 0, -1], index=self.idx10)

        pd.testing.assert_frame_equal(test6_actual, test6)
        pd.testing.assert_series_equal(test7_actual, test7, check_names=False)

    def test_with_standardization(self):
        """
        Test cases with standardization, with constant and dynamic threshold.
        """
        close = self.data[['SPY', 'EPP', 'FXI']][:10]
        threshold2 = pd.Series([1, 2, 0, 0.2, 0.02, 1.5, 10, -1, 500, 1], index=self.idx10)

        test8 = fixed_time_horizon(close, threshold=1, lag=False, standardized=True, window=4)
        test9 = fixed_time_horizon(close, threshold=0.1, lag=True, standardized=True, window=5)
        test10 = fixed_time_horizon(close, threshold=threshold2, lag=True, standardized=True, window=3)
        test8_actual = pd.DataFrame({'SPY': [np.nan, np.nan, np.nan, np.nan, 0, 1, 0, 0, 0, -1],
                                     'EPP': [np.nan, np.nan, np.nan, np.nan, 0, 1, 0, -1, 0, -1],
                                     'FXI': [np.nan, np.nan, np.nan, np.nan, 0, 1, 0, -1, 0, -1]},
                                    index=self.idx10)
        test9_actual = pd.DataFrame({'SPY': [np.nan, np.nan, np.nan, np.nan, 1, 1, -1, 1, -1, np.nan],
                                     'EPP': [np.nan, np.nan, np.nan, np.nan, 1, 1, -1, 1, -1, np.nan],
                                     'FXI': [np.nan, np.nan, np.nan, np.nan, 1, -1, -1, 0, -1, np.nan]},
                                    index=self.idx10)
        test10_actual = pd.DataFrame({'SPY': [np.nan, np.nan, 1, 0, 1, 0, 0, 1, 0, np.nan],
                                      'EPP': [np.nan, np.nan, 1, -1, 1, 0, 0, 1, 0, np.nan],
                                      'FXI': [np.nan, np.nan, 1, -1, 1, 0, 0, 1, 0, np.nan]},
                                     index=self.idx10)

        pd.testing.assert_frame_equal(test8_actual, test8)
        pd.testing.assert_frame_equal(test9_actual, test9)
        pd.testing.assert_frame_equal(test10_actual, test10)

    def test_resample(self):
        """
        Tests for when a resample period is used.
        """
        cols = ['SPY', 'EPP', 'FXI']
        close1 = self.data[cols].iloc[0:30]
        close2 = self.data[cols].iloc[0:150]
        week_index = close1.resample('W').last().index
        month_index = close2.resample('M').last().index
        threshold3 = pd.Series([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], index=month_index)

        test11 = fixed_time_horizon(close1, threshold=0.02, resample_by='W', lag=True, standardized=False)
        test12 = fixed_time_horizon(close2, threshold=threshold3, resample_by='M', lag=True, standardized=True, window=3)
        test11_actual = pd.DataFrame({'SPY': [0, -1, 0, 1, -1, 1, np.nan],
                                      'EPP': [0, -1, 0, 1, -1, 0, np.nan],
                                      'FXI': [1, -1, -1, 0, -1, 1, np.nan]},
                                     index=week_index)
        test12_actual = pd.DataFrame({'SPY': [np.nan, np.nan, 1, 0, -1, 0, 0, np.nan],
                                      'EPP': [np.nan, np.nan, 1, 0, -1, 0, 0, np.nan],
                                      'FXI': [np.nan, np.nan, 1, 0, -1, 1, 0, np.nan]},
                                     index=month_index)

        pd.testing.assert_frame_equal(test11_actual, test11)
        pd.testing.assert_frame_equal(test12_actual, test12)

    def test_exceptions_warnings(self):
        """
        Tests the exceptions and warning that can be raised.
        """
        close = self.data[['SPY', 'EWG']][:10]
        threshold = pd.Series([0.01]*10)
        with self.assertRaises(Exception):  # Threshold index doesn't match.
            fixed_time_horizon(close, threshold)
        with self.assertRaises(Exception):  # Standardized but no window.
            fixed_time_horizon(close, 0.01, standardized=True)
        with self.assertWarns(UserWarning):  # Window too long.
            fixed_time_horizon(close, 0.01, standardized=True, window=50)
