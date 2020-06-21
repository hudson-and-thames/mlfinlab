# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.labeling.excess_over_mean import excess_over_mean


class TestLabelingOverMean(unittest.TestCase):
    """
    Tests regarding labeling excess over median.
    """

    def setUp(self):
        """
        Set the file path for the sample data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.data.index = pd.to_datetime(self.data.index)

    def test_small_set(self):
        """
        Check for a small set with manually inputted results, with numerical and categorical outputs.
        """
        cols = ['EEM', 'EWG', 'TIP', 'EWJ']
        subset = self.data[cols].iloc[0:5]
        test1 = excess_over_mean(subset)
        test2 = excess_over_mean(subset, binary=True)
        test1_actual = pd.DataFrame([(0.005666, -0.006157, 0.00004411, 0.0004476),
                                     (-0.011169, -0.000684, 0.018588, -0.006734),
                                     (0.006871, 0.000411, -0.000643, -0.006639),
                                     (-0.003687, -0.002863, 0.004815, 0.001735),
                                     (np.nan, np.nan, np.nan, np.nan), ], columns=self.data[cols].iloc[0:5].columns,
                                    index=self.data[cols].iloc[0:5].index)
        test2_actual = test1_actual.apply(np.sign)

        # Check less precise because calculated numbers have more decimal places than inputted ones.
        pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test2, test2_actual)

    def test_large_set(self):
        """
        Checks a specific row for a large dataset, and ensures the last row is NaN.
        """
        # Get a random row of the entire dataset.
        test3 = pd.Series(excess_over_mean(self.data, lag=True).iloc[42])
        idx42 = self.data.iloc[42].index
        test3_actual = pd.Series([0.014267, 0.00537, -0.007375, -0.001013, 0.005282, -0.011075, 0.006921, 0.000626,
                                  0.021376, 0.018429, -0.01463, -0.01377, 0.00048, -0.009569, 0.0107, 0.0121335,
                                  0.0050173, 0.0002504, -0.0004207, -0.018707, -0.013208, -0.00818807, -0.002897],
                                 index=idx42)

        pd.testing.assert_series_equal(test3, test3_actual, check_less_precise=True, check_names=False)

        test4 = excess_over_mean(self.data, lag=True).iloc[-1]
        self.assertTrue(test4.isnull().all())

    def test_resample_period(self):
        """
        Test numerical and categorical labels with a resample period.
        """
        cols = ['EEM', 'EWG', 'TIP', 'EWJ']
        subset1 = self.data[cols].iloc[0:25]
        subset2 = self.data[cols].iloc[0:100]
        week_idx = subset1.resample('W').last().index
        month_idx = subset2.resample('M').last().index
        # Resample per week and per month.
        test5 = excess_over_mean(subset1, binary=False, resample_by='W', lag=True)
        test6 = excess_over_mean(subset2, binary=False, resample_by='M', lag=False)
        test7 = excess_over_mean(subset2, binary=True, resample_by='M', lag=False)

        test5_actual = pd.DataFrame({'EEM': [0.017255, -0.042112, 0.004907, 0.016267, -0.026054, np.nan],
                                     'EWG': [-0.011975, -0.019257, -0.048906, 0.017310, -0.003467, np.nan],
                                     'TIP': [0.004898, 0.047419, 0.026425, -0.031773, 0.038212, np.nan],
                                     'EWJ': [-0.010178, 0.013950, 0.017574, -0.001804, -0.008691, np.nan]},
                                    index=week_idx)
        test6_actual = pd.DataFrame({'EEM': [np.nan, 0.018196, -0.027718, 0.047210, 0.001358],
                                     'EWG': [np.nan, -0.008799, 0.026921, -0.007042, 0.007712],
                                     'TIP': [np.nan, 0.0072872, 0.0037534, -0.069560, 0.0004006],
                                     'EWJ': [np.nan, -0.016684, -0.002956, 0.029392, -0.009471]},
                                    index=month_idx)

        pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test7, test6_actual.apply(np.sign))
