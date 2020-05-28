# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.labeling.excess_over_median import excess_over_median


class TestLabellingOverMedian(unittest.TestCase):
    """
    Tests regarding labelling excess over median.
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
        cols = ['EEM', 'EWG', 'TIP']
        subset = self.data[cols].iloc[0:5]
        test1 = excess_over_median(subset)
        test2 = excess_over_median(subset, binary=True)
        test1_actual = pd.DataFrame([(0.005622, -0.006201, 0), (-0.010485, 0, 0.019272), (0.006460, 0, -0.001054),
                                     (-0.000824, 0, 0.007678), (np.nan, np.nan, np.nan), ], columns=self.data[cols].
                                    iloc[0:5].columns, index=self.data[cols].iloc[0:5].index)
        test2_actual = np.sign(test1_actual)

        # Check less precise because calculated numbers have more decimal places than inputted ones
        pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test2, test2_actual)

    def test_last_row(self):
        """
        Test for a larger dataset. Verifies that the last row is NaN.
        """
        test3 = excess_over_median(self.data)
        test4 = excess_over_median(self.data.iloc[:, 0:20], binary=True)
        # Verify that last row is NaN
        self.assertTrue(test3.iloc[-1].isnull().all())
        self.assertTrue(test4.iloc[-1].isnull().all())

    def test_shape(self):
        """
        Since the returns are compared to the median, there should in theory be an equal number of +1 and -1 labels.
        However, if the dataset is large, there is a higher chance that multiple tickers may have the same exact median
        value, so the number of +1 and -1 labels won't match exactly. This test checks if the number of those labels are
        within 2% of each other.
        """
        test5 = excess_over_median(self.data, binary=True)
        num_negative_ones = test5.stack().value_counts()[-1]
        num_positive_ones = test5.stack().value_counts()[1]
        self.assertTrue(0.98 * num_negative_ones < num_positive_ones < 1.02 * num_negative_ones)
