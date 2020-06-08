# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.labeling.raw_return import raw_return


class TestLabelingRawReturns(unittest.TestCase):
    """
    Tests for the raw returns labeling method.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.idx5 = self.data[:5].index
        self.col5 = self.data.iloc[:, 0:5].columns

    def test_dataframe(self):
        """
        Verifies raw returns for a DataFrame.
        """
        price = self.data.iloc[0:5, 0:5]
        test1 = raw_return(price)
        test1_actual = pd.DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan),
                                     (0.008997, -0.002826, 0.0033758, 0.003779, 0.001662),
                                     (-0.030037, -0.019552, -0.0002804, -0.025602, -0.022719),
                                     (0.007327, 0.000867, -0.000187, -0.006182, 0.001045),
                                     (-0.007754, -0.006930, 0.000748, -0.002333, -0.005610)],
                                    columns=self.col5, index=self.idx5)

        pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)

    def test_series(self):
        """
        Verifies raw returns for a series for percentage/logarithmic returns, with numerical/binary labels.
        """
        # Takes series of 10 imaginary prices
        price = pd.Series([100, 101, 102, 102, 102, 99, 19, 2000, 100, 105])

        test1 = raw_return(price)
        test2 = raw_return(price, binary=True)
        test3 = raw_return(price, logarithmic=True)
        test4 = raw_return(price, binary=True, logarithmic=True)
        test5 = raw_return(price, binary=False, logarithmic=False, lookback=3)
        test6 = raw_return(price, binary=False, logarithmic=True, lookback=4)
        test1_actual = pd.Series([np.nan, 0.01, 0.009901, 0, 0, -0.029412, -0.808081, 104.263158, -0.95, 0.05])
        test3_actual = pd.Series([np.nan, 0.00995, 0.009852, 0, 0, -0.029853, -1.650681, 4.656463, -2.995732, 0.048790])
        test5_actual = pd.Series([np.nan, np.nan, np.nan, 0.02, 0.009901, -0.029412, -0.813725, 18.607843, 0.010101,
                                  4.526316])
        test6_actual = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.019803, -0.020001, -1.680534, 2.975930, -0.019803,
                                  0.058841])

        # Check less precise for slight rounding differences between calculated and manually inputted values
        pd.testing.assert_series_equal(test1, test1_actual, check_less_precise=True)
        self.assertEqual(test2.all(), test1.apply(np.sign).all())
        pd.testing.assert_series_equal(test3, test3_actual, check_less_precise=True)
        self.assertEqual(test4.all(), test3.apply(np.sign).all())
        pd.testing.assert_series_equal(test5, test5_actual, check_less_precise=True)
        pd.testing.assert_series_equal(test6, test6_actual, check_less_precise=True)

    def test_warning(self):
        """
        Test for warning when lookback is greater than number of rows.
        """
        price = self.data[:10]
        with self.assertWarns(UserWarning):
            raw_return(price, lookback=999)
