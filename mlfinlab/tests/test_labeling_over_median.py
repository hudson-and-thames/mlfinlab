# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.excess_over_median import excess_over_median


class TestLabellingOverMedian(unittest.TestCase):
    """
    Tests regarding labelling excess over median
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
        Check for a small manually set with manually inputted results, both binary both off and on
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
        Test for a larger dataset. Verifies that the last row is correctly NaN and that 0 labels are given correctly.
        """
        test3 = excess_over_median(self.data)
        test4 = excess_over_median(self.data.iloc[:, 0:20], binary=True)

        # Verify that last row is NaN
        self.assertTrue(test3.iloc[-1].isnull().all())
        self.assertTrue(test4.iloc[-1].isnull().all())

