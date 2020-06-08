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
        self.data = pd.read_csv(self.path, index_col='Date')
        self.idx10 = self.data[:10].index

    def test_basic(self):
        """
        Tests for the basic case where the benchmark is a constant.
        """
        data1 = self.data['EWQ']
        data3 = self.data[['EWU', 'XLB']]

        # No benchmark given, assumed to be 0
        test1 = return_over_benchmark(data1[:10])
        test2 = return_over_benchmark(data1[:10], binary=True)

        # Constant benchmark, on multiple columns
        test3 = return_over_benchmark(data3[:10], 0.005)

        test1_actual = pd.Series([np.nan, -0.00052716, -0.02452523, 0.00729918, -0.00778307, 0.0029754, 0.00242709,
                                  -0.0191014, 0.02057049, -0.02983071], index=self.idx10)
        test3_actual = pd.DataFrame([[np.nan, np.nan], [0.001281, 0.013160], [-0.028720, -0.035202],
                                     [-0.002016, -0.018732], [-0.019025, -0.020415], [-0.004569, 0.001313],
                                     [-0.010170, 0.008551], [-0.026221, -0.015151], [0.006947, 0.028517],
                                     [-0.029924, -0.032348]], index=self.idx10, columns=data3.columns)
        pd.testing.assert_series_equal(test1, test1_actual, check_names=False)
        pd.testing.assert_series_equal(test2, test1_actual.apply(np.sign), check_names=False)
        pd.testing.assert_frame_equal(test3, test3_actual, check_names=False, check_less_precise=True)

    def test_given_benchmark(self):
        """
        Tests comparing value to a given benchmark
        """
        # User inputted benchmark
        benchmark4 = pd.Series([0, 0.01, -0.01, 0.02, -0.005, 0.6, 100, -90, -0.2, 0.008], index=self.idx10)
        data4 = self.data['BND']
        test4 = return_over_benchmark(data4[:10], benchmark4)
        test5 = return_over_benchmark(data4[:10], benchmark4, binary=True)
        test4_actual = pd.Series([np.nan, -8.70736203e-03, 1.11619412e-02, -1.97421452e-02, 6.03135015e-03,
                                  -6.01159098e-01, -1.00000387e+02, 9.00030955e+01, 2.01285921e-01,
                                  -4.40427932e-03], index=self.idx10)

        # Using SPY as a benchmark
        benchmark6 = self.data['SPY'].pct_change(periods=1)
        test6 = return_over_benchmark(data4[:10], benchmark6[:10])
        test6_actual = pd.Series([np.nan, 0.00177558, 0.02566838, 0.00110702, 0.01717979, -0.01166944, -0.00694088,
                                  0.01116406, -0.0067769, 0.02560875], index=self.idx10)

        pd.testing.assert_series_equal(test4, test4_actual)
        pd.testing.assert_series_equal(test5, test4_actual.apply(np.sign))
        pd.testing.assert_series_equal(test6, test6_actual)

    def test_warning(self):
        """
        Verifies that the correct warning is given when there is a mismatch between
        """
        returns = self.data['TLT'].pct_change()
        benchmark = self.data['SPY'].pct_change()

        # Case where benchmark is 2141 in length while price is only 10
        with self.assertWarns(UserWarning):
            return_over_benchmark(returns[:10], benchmark)
