# pylint: disable=missing-module-docstring

import unittest
import os
from datetime import datetime
import numpy as np
import pandas as pd


from mlfinlab.labeling.excess_over_median import excess_over_median


class TestLabelingOverMedian(unittest.TestCase):
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

    def test_basic(self):
        """
        Test basic case for a small set with manually inputted results, with numerical and categorical outputs, with
        no resampling or forward looking labels.
        """
        cols = ['EEM', 'EWG', 'TIP']
        subset = self.data[cols].iloc[0:5]
        test1 = excess_over_median(subset, lag=False)
        test2 = excess_over_median(subset, binary=True, lag=False)
        test1_actual = pd.DataFrame([(np.nan, np.nan, np.nan), (0.0056216, -0.006201, 0), (-0.010485, 0, 0.019272),
                                     (0.006460, 0, -0.001054), (-0.000824, 0, 0.007678)],
                                    columns=cols, index=self.data[cols].iloc[0:5].index)
        test2_actual = test1_actual.apply(np.sign)

        # Check less precise because calculated numbers have more decimal places than inputted ones.
        pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test2, test2_actual)

    def test_resample_period(self):
        """
        Test numerical and categorical with a resample period.
        """
        cols = ['EEM', 'EWG', 'TIP', 'EWJ']
        subset1 = self.data[cols].iloc[0:25]
        subset2 = self.data[cols].iloc[0:100]

        # Resample per business day. Should be same as no resampling at all (after removing Jan 21, since Python
        # considers that a business day even though it was MLK day during the year of the data, so no trading occurred).
        test3 = excess_over_median(subset1, resample_by='B', lag=False)
        test3.drop(datetime.strptime('2008-01-21', '%Y-%m-%d'), inplace=True)  # MLK day, 2008
        pd.testing.assert_frame_equal(test3, excess_over_median(subset1, lag=False))

        # Resample per week and month.
        test4 = excess_over_median(subset1, resample_by='W', lag=False)
        weekly_index = subset1.resample('W').last().index
        test5 = excess_over_median(subset2, resample_by='M', lag=False)
        monthly_index = subset2.resample('M').last().index
        test6 = excess_over_median(subset2, binary=True, resample_by='M', lag=False)  # Binary by month

        test4_actual = pd.DataFrame([(np.nan, np.nan, np.nan, np.nan),
                                     (0.019896, -0.009335, 0.007538, -0.007538),
                                     (-0.039458, -0.016603, 0.050073, 0.016603),
                                     (-0.006333, -0.060147, 0.015185, 0.006333),
                                     (0.009036, 0.010079, -0.039004, -0.009036),
                                     (-0.019975, 0.002612, 0.044291, -0.002612)],
                                    columns=cols, index=weekly_index)
        test5_actual = pd.DataFrame([(np.nan, np.nan, np.nan, np.nan),
                                     (0.018951, -0.008043, 0.008043, -0.015929),
                                     (-0.028116, 0.026522, 0.003355, -0.003355),
                                     (0.036035, -0.018217, -0.080735, 0.018217),
                                     (0.000479, 0.006833, -0.000479, -0.010351)],
                                    columns=cols, index=monthly_index)

        pd.testing.assert_frame_equal(test4, test4_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test5, test5_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test6, test5_actual.apply(np.sign), check_less_precise=True)

    def test_forward(self):
        """
        Tests with lagged returns.
        """
        cols = ['EEM', 'EWG', 'TIP']
        subset = self.data[cols].iloc[0:5]
        subset2 = self.data[cols].iloc[0:100]
        monthly_index = subset2.resample('M').last().index

        test7 = excess_over_median(subset, lag=True)
        test8 = excess_over_median(subset, binary=True, lag=True)
        test9 = excess_over_median(subset2, resample_by='M', lag=True)
        test10 = excess_over_median(subset2, resample_by='M', binary=True, lag=True)

        test7_actual = pd.DataFrame([(0.0056216, -0.006201, 0), (-0.010485, 0, 0.019272), (0.006460, 0, -0.001054),
                                     (-0.000824, 0, 0.007678), (np.nan, np.nan, np.nan)],
                                    columns=cols, index=self.data[cols].iloc[0:5].index)
        test9_actual = pd.DataFrame([(0.010909, -0.016086, 0), (-0.031471, 0.023167, 0), (0.054252, 0, -0.062518),
                                     (0, 0.006354, -0.000958), (np.nan, np.nan, np.nan)],
                                    columns=cols, index=monthly_index)

        pd.testing.assert_frame_equal(test7, test7_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign), check_less_precise=True)
        pd.testing.assert_frame_equal(test9, test9_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test10, test9_actual.apply(np.sign), check_less_precise=True)

    def test_nan(self):
        """
        Tests to check that NaN values in prices get ignored.
        """
        cols = ['EEM', 'EWG', 'TIP']
        subset = self.data[cols].iloc[0:5]
        with_nan = pd.concat([subset, pd.Series([np.nan]*5, name='nan', index=subset.index)], axis=1)
        test11 = excess_over_median(with_nan)
        test11.drop('nan', axis=1, inplace=True)
        pd.testing.assert_frame_equal(test11, excess_over_median(subset), check_less_precise=True)
