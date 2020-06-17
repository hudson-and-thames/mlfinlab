# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.return_vs_benchmark import return_over_benchmark


class TestReturnOverBenchmark(unittest.TestCase):
    """
    Tests regarding the labeling returns over benchmark method.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)
        self.idx10 = self.data[:10].index

    def test_basic(self):
        """
        Tests for the basic case where the benchmark is a constant.
        """
        data1 = self.data['EWQ']
        data3 = self.data[['EWU', 'XLB']]

        # No benchmark given, assumed to be 0.
        test1 = return_over_benchmark(data1[:10], lag=True)
        test2 = return_over_benchmark(data1[:10], binary=True, lag=True)

        # Constant benchmark, on multiple columns.
        test3 = return_over_benchmark(data3[:10], benchmark=0.005, lag=False)

        test1_actual = pd.Series([-0.00052716, -0.02452523, 0.00729918, -0.00778307, 0.0029754, 0.00242709,
                                  -0.0191014, 0.02057049, -0.02983071, np.nan], index=self.idx10)
        test3_actual = pd.DataFrame({'EWU': [np.nan, 0.001281, -0.028720, -0.002016, -0.019025, -0.004569, -0.010170,
                                             -0.026221, 0.006947, -0.029924],
                                     'XLB': [np.nan, 0.013160, -0.035202, -0.018732, -0.020415, 0.001313, 0.008551,
                                             -0.015151, 0.028517, -0.032348]},
                                    index=self.idx10, columns=data3.columns)
        pd.testing.assert_series_equal(test1, test1_actual, check_names=False)
        pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)
        pd.testing.assert_frame_equal(test3, test3_actual, check_less_precise=True)

    def test_given_benchmark(self):
        """
        Tests comparing value to a dynamic benchmark.
        """
        # User inputted benchmark.
        benchmark4 = pd.Series([0, 0.01, -0.01, 0.02, -0.005, 0.6, 100, -90, -0.2, 0.008], index=self.idx10)
        data4 = self.data['BND']
        test4 = return_over_benchmark(data4[:10], benchmark=benchmark4, lag=False)
        test5 = return_over_benchmark(data4[:10], benchmark=benchmark4, binary=True, lag=False)
        test4_actual = pd.Series([np.nan, -8.70736203e-03, 1.11619412e-02, -1.97421452e-02, 6.03135015e-03,
                                  -6.01159098e-01, -1.00000387e+02, 9.00030955e+01, 2.01285921e-01,
                                  -4.40427932e-03], index=self.idx10)

        # Using SPY as a benchmark.
        benchmark6 = self.data['SPY'].pct_change(periods=1)
        test6 = return_over_benchmark(data4[:10], benchmark6[:10], lag=False)
        test6_actual = pd.Series([np.nan, 0.00177558, 0.02566838, 0.00110702, 0.01717979, -0.01166944, -0.00694088,
                                  0.01116406, -0.0067769, 0.02560875], index=self.idx10)

        pd.testing.assert_series_equal(test4, test4_actual)
        pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))
        pd.testing.assert_series_equal(test6, test6_actual)

    def test_resample(self):
        """
        Tests for when resampling is used.
        """
        data5 = self.data[['EEM', 'EWG', 'TIP']]
        subset1 = data5[40:50]
        subset2 = data5[0:130]
        benchmark_day = pd.Series([0.01, 0.01, 0.01, -0.01, -0.01, -0.02, 0.2, 0.04, -0.1, 0], index=subset1.index)
        month_index = subset2.resample('M').last().index

        test7 = return_over_benchmark(subset1, benchmark=benchmark_day, binary=False, resample_by='B', lag=True)
        test7b = return_over_benchmark(subset1, benchmark=benchmark_day, binary=False, lag=True)
        test8 = return_over_benchmark(subset2, benchmark=-0.02, binary=True, resample_by='M', lag=True)  # Negative
        test8_actual = pd.DataFrame({'EEM': [1, -1, 1, 1, -1, -1, np.nan], 'EWG': [1, 1, 1, 1, -1, 1, np.nan],
                                     'TIP': [1, 1, -1, 1, 1, 1, np.nan]}, index=month_index)

        pd.testing.assert_frame_equal(test7, test7b)
        pd.testing.assert_frame_equal(test8, test8_actual)

    def test_exception(self):
        """
        Verifies that the exception is given when there is a mismatch between prices.index and benchmark.index.
        """
        returns = self.data['TLT'].pct_change()
        benchmark = self.data['SPY'].pct_change()

        # Suppose we resample the returns, but fail to update the index of benchmark.
        with self.assertRaises(Exception):
            return_over_benchmark(returns, benchmark=benchmark, resample_by='W')
