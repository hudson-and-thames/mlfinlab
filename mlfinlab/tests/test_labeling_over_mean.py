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

        # Check less precise because calculated numbers have more decimal places than inputted ones
        pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test2, test2_actual)

    def test_large_set(self):
        """
        Checks a specific row for a large dataset, and ensures the last row is NaN.
        """
        # Get a random row of the entire dataset
        test3 = pd.Series(excess_over_mean(self.data).iloc[42])
        idx42 = self.data.iloc[42].index
        test3_actual = pd.Series([0.014267, 0.00537, -0.007375, -0.001013, 0.005282, -0.011075, 0.006921, 0.000626,
                                  0.021376, 0.018429, -0.01463, -0.01377, 0.00048, -0.009569, 0.0107, 0.0121335,
                                  0.0050173, 0.0002504, -0.0004207, -0.018707, -0.013208, -0.00818807, -0.002897],
                                 index=idx42)

        pd.testing.assert_series_equal(test3, test3_actual, check_less_precise=True, check_names=False)

        test4 = excess_over_mean(self.data).iloc[-1]
        self.assertTrue(test4.isnull().all())
