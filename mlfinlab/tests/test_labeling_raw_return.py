# pylint: disable=missing-module-docstring
# pylint: disable=no-self-use

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
        self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
        self.idx5 = self.data[:5].index
        self.col5 = self.data.iloc[:, 0:5].columns

    def test_dataframe(self):
        """
        Verifies raw returns for a DataFrame.
        """
        prices = self.data.iloc[0:5, 0:5]
        test1 = raw_return(prices, lag=False)
        test2 = raw_return(prices, binary=True, lag=False)
        test3 = raw_return(prices, logarithmic=True, lag=True)
        test4 = raw_return(prices, binary=True, logarithmic=True, lag=True)
        test1_actual = pd.DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan),
                                     (0.008997, -0.002826, 0.0033758, 0.003779, 0.001662),
                                     (-0.030037, -0.019552, -0.0002804, -0.025602, -0.022719),
                                     (0.007327, 0.000867, -0.000187, -0.006182, 0.001045),
                                     (-0.007754, -0.006930, 0.000748, -0.002333, -0.005610)],
                                    columns=self.col5, index=self.idx5)

        test3_actual = pd.DataFrame([(0.008957, -0.002830, 0.003370, 0.003772, 0.001661),
                                     (-0.030497, -0.019746, -0.0002804, -0.025936, -0.022981),
                                     (0.007300, 0.000867, -0.000187, -0.006202, 0.001044),
                                     (-0.007784, -0.006955, 0.0007478, -0.002336, -0.005626),
                                     (np.nan, np.nan, np.nan, np.nan, np.nan)],
                                    columns=self.col5, index=self.idx5)

        pd.testing.assert_frame_equal(test1, test1_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test2, test1_actual.apply(np.sign))
        pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test4, test3_actual.apply(np.sign))

    def test_series(self):
        """
        Verifies raw returns for a series for simple/logarithmic returns, with numerical/binary labels.
        """
        # Takes series of 10 imaginary prices.
        price = pd.Series([100, 101, 102, 102, 102, 99, 19, 2000, 100, 105])

        test4 = raw_return(price, lag=True)
        test5 = raw_return(price, logarithmic=True, lag=False)
        test6 = raw_return(price, binary=True, logarithmic=True, lag=False)
        test4_actual = pd.Series([0.01, 0.009901, 0, 0, -0.029412, -0.808081, 104.263158, -0.95, 0.05, np.nan])
        test5_actual = pd.Series([np.nan, 0.00995033, 0.0098523, 0, 0, -0.02985296, -1.65068087, 4.65646348,
                                  -2.99573227, 0.04879016])
        pd.testing.assert_series_equal(test4, test4_actual, check_less_precise=True)
        pd.testing.assert_series_equal(test5, test5_actual, check_less_precise=True)
        pd.testing.assert_series_equal(test6, test5_actual.apply(np.sign))

    def test_resample(self):
        """
        Tests that resampling works correctly.
        """
        price1 = self.data.iloc[0:25, 0:5]
        price2 = self.data.iloc[:, 0:3]
        week_index = price1.resample('W').last().index
        year_index = self.data.resample('Y').last().index
        test6 = raw_return(price1, binary=False, logarithmic=True, resample_by='W', lag=True)
        test7 = raw_return(price2, binary=False, logarithmic=False, resample_by='Y', lag=True)
        test8 = raw_return(price2, binary=True, logarithmic=False, resample_by='Y', lag=True)
        test6_actual = pd.DataFrame([(0.014956, -0.014263, 0.002707, -0.012442, -0.018586),
                                     (-0.081178, -0.056693, 0.011494, -0.022153, -0.046989),
                                     (-0.015181, -0.071367, 0.006431, -0.002403, -0.014326),
                                     (0.045872, 0.046868, -0.001099, 0.028460, 0.039661),
                                     (-0.068778, -0.044871, -0.002203, -0.050350, -0.058079),
                                     (np.nan, np.nan, np.nan, np.nan, np.nan)],
                                    columns=self.col5, index=week_index)
        test7_actual = pd.DataFrame({'EEM': [0.661994, 0.147952, -0.203610, 0.168951, -0.057497, -0.060048, -0.180708,
                                             0.077664, np.nan],
                                     'EWG': [0.167534, 0.066845, -0.197160, 0.285120, 0.285830, -0.136965, -0.044509,
                                             -0.079038, np.nan],
                                     'TIP': [0.046957, 0.034841, 0.085287, 0.040449, -0.094803, 0.019199, -0.020802,
                                             0.067287, np.nan]}, index=year_index)
        pd.testing.assert_frame_equal(test6, test6_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test7, test7_actual, check_less_precise=True)
        pd.testing.assert_frame_equal(test8, test7_actual.apply(np.sign))
