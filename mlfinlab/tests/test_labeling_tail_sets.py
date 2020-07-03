# pylint: disable=missing-module-docstring
# pylint: disable=protected-access

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.tail_sets import TailSetLabels


class TestTailSets(unittest.TestCase):
    """
    Unit tests for the tail sets labeling class.
    """

    def setUp(self):
        """
        Set the file path for the sample data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date', parse_dates=True)

    def test_init(self):
        """
        Tests to ensure exceptions are correctly raised for invalid inputs.
        """
        # Too large n_bins.
        with self.assertRaises(Exception):
            TailSetLabels(self.data[:100], n_bins=200)
        # Window isn't int with vol_adj.
        with self.assertRaises(Exception):
            TailSetLabels(self.data[:100], n_bins=10, vol_adj='stdev', window='str')
        # Window is too large
            TailSetLabels(self.data[:100], n_bins=10, vol_adj='stdev', window=200)

    def test_vol_adj_ret(self):
        """
        Tests for getting the correct volatility adjusted return.
        """
        # Check for no vol adjustment. Should be same as straight log returns.
        label1 = TailSetLabels(self.data[:100], n_bins=23, vol_adj=None, window=None)  # No vol adjustment
        test1 = label1.vol_adj_rets
        simple_returns = np.log(self.data[:100]).diff().dropna()

        # # Stdev adjusted returns.
        label2 = TailSetLabels(self.data[:100], n_bins=8, vol_adj='stdev', window=20)
        test2 = label2.vol_adj_rets
        test2_actual = (simple_returns / simple_returns.rolling(20).std()).dropna()

        # Mean abs returns
        label3 = TailSetLabels(self.data[:100], n_bins=12, vol_adj='mean_abs_dev', window=20)
        test3 = label3.vol_adj_rets
        pd.testing.assert_frame_equal(test1, simple_returns)
        pd.testing.assert_frame_equal(test2, test2_actual)
        np.testing.assert_array_almost_equal(test3.iloc[4:5, 3:13],
                                             np.array([[-0.88220253, -0.05699642, -0.41151834, 0.22209753, -0.26852039,
                                                        -0.41058931, -0.72457246, -0.45304492, -1.62358571,
                                                        -0.74637241]]))

    def test_extract_tail_sets(self):
        """
        Tests for extracting the tail set in one row, including positive and negative class.
        """
        label4 = TailSetLabels(self.data[:100], n_bins=4, vol_adj='mean_abs_dev', window=20)
        returns4 = label4.vol_adj_rets
        test4 = label4._extract_tail_sets(row=returns4.iloc[25])
        test4_actual = pd.Series([0, 0, 1, -1, 0, 1, 0, 0, -1, -1, 1, 0, 0, 0, -1, 0, 0, -1, 0, 1, 1, 1, -1],
                                 index=returns4.iloc[25].index)
        test5 = label4._positive_tail_set(test4)
        test5_actual = ['TIP', 'IEF', 'XLF', 'TLT', 'BND', 'CSJ']
        test6 = label4._negative_tail_set(test4)
        test6_actual = ['EWJ', 'XLB', 'XLE', 'EPP', 'VPL', 'DIA']
        pd.testing.assert_series_equal(test4, test4_actual)
        self.assertEqual(test5, test5_actual)
        self.assertEqual(test6, test6_actual)

    def test_overall(self):
        """
        Tests the overall output of the tail set labels.
        """
        label7 = TailSetLabels(self.data[:100], n_bins=10, vol_adj='mean_abs_dev', window=30)
        test7_pos, test7_neg, _ = label7.get_tail_sets()

        self.assertEqual(test7_pos[1], ['XLU', 'EPP', 'FXI'])
        self.assertEqual(test7_neg[1], ['EWU', 'XLK', 'DIA'])
