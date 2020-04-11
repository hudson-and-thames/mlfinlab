"""
Test Trend Scanning labels
"""

import os
import unittest
import pandas as pd

from mlfinlab.labeling import trend_scanning_labels


class TestTrendScanningLabels(unittest.TestCase):
    """
    Test trend-scanning labels
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)

        # Data set used for trend scanning labels
        self.eem_close = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0, parse_dates=[0])
        # In 2008, EEM had some clear trends
        self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

    def test_trend_scanning_labels(self):
        """
        Test trend scanning labels
        """

        t_events = self.eem_close.index
        tr_scan_labels = trend_scanning_labels(self.eem_close, t_events, 20)

        self.assertEqual(tr_scan_labels.shape[0], len(t_events))  # we have label value for all t events

        # Before 2008/5/12 we had a strong positive trend
        self.assertTrue(
            set(tr_scan_labels.loc[pd.Timestamp(2008, 1, 1):pd.Timestamp(2008, 5, 9)].bin) == set([1]))

        self.assertEqual(tr_scan_labels.bin.value_counts()[-1], 70)  # Number of -1 labels check
        self.assertEqual(tr_scan_labels.bin.value_counts()[1], 40)

        # Approx. 20 latest labels are nan because we don't have data for look-forward period (20)
        self.assertEqual(tr_scan_labels.shape[0] - tr_scan_labels.dropna().shape[0], 19)

        tr_scan_labels.dropna(inplace=True)  # Drop na values

        # Index should be < t1
        self.assertTrue((tr_scan_labels.t1 > tr_scan_labels.index).all())

        for int_index, (ret_v, bin_v) in zip([0, 2, 10, 20, 50],
                                             [(0.05037, 1), (0.0350, 1), (0.07508, 1), (0.05219, 1), (0.02447, 1)]):
            self.assertAlmostEqual(tr_scan_labels.iloc[int_index]['ret'], ret_v, delta=1e-4)
            self.assertEqual(tr_scan_labels.iloc[int_index]['bin'], bin_v)

        tr_scan_labels_none = trend_scanning_labels(self.eem_close, t_events=None, look_forward_window=20)
        tr_scan_labels_none.dropna(inplace=True)

        self.assertTrue((tr_scan_labels == tr_scan_labels_none).all().all())
