"""
Test various functions regarding chapter 4: Return/Time attribution
"""

import os
import unittest

import pandas as pd

from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier
from mlfinlab.sample_weights.attribution import get_weights_by_return, get_weights_by_time_decay
from mlfinlab.util.volatility import get_daily_vol


class TestSampling(unittest.TestCase):
    """
    Test Triple barrier, meta-labeling, dropping rare labels, and daily volatility.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data and get triple barrier events
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                 num_days=2)

        self.data['side'] = 1
        self.meta_labeled_events = get_events(close=self.data['close'],
                                              t_events=cusum_events,
                                              pt_sl=[4, 4],
                                              target=daily_vol,
                                              min_ret=0.005,
                                              num_threads=3,
                                              vertical_barrier_times=vertical_barriers,
                                              side_prediction=self.data['side'])

    def test_ret_attribution(self):
        """
        Assert that return attribution length equals triple barrier length, check particular values
        """
        non_nan_meta_labels = self.meta_labeled_events.dropna()
        ret_weights = get_weights_by_return(non_nan_meta_labels, self.data['close'])
        self.assertTrue(ret_weights.shape[0] == non_nan_meta_labels.shape[0])
        self.assertTrue(abs(ret_weights.iloc[0] - 0.781807) <= 1e5)
        self.assertTrue(abs(ret_weights.iloc[3] - 1.627944) <= 1e5)

    def test_time_decay_weights(self):
        """
        Assert that time decay weights length equals triple barrier length, check particular values
        """
        non_nan_meta_labels = self.meta_labeled_events.dropna()
        standard_decay = get_weights_by_time_decay(non_nan_meta_labels, self.data['close'], decay=0.5)
        no_decay = get_weights_by_time_decay(non_nan_meta_labels, self.data['close'], decay=1)
        neg_decay = get_weights_by_time_decay(non_nan_meta_labels, self.data['close'], decay=-0.5)
        converge_decay = get_weights_by_time_decay(non_nan_meta_labels, self.data['close'], decay=0)
        pos_decay = get_weights_by_time_decay(non_nan_meta_labels, self.data['close'], decay=1.5)

        self.assertTrue(standard_decay.shape == no_decay.shape)
        self.assertTrue(standard_decay.shape == neg_decay.shape)
        self.assertTrue(standard_decay.shape == converge_decay.shape)
        self.assertTrue(standard_decay.shape == pos_decay.shape)

        self.assertTrue(standard_decay.iloc[-1] == 1.0)
        self.assertTrue(abs(standard_decay.iloc[0] - 0.582191) <= 1e5)

        self.assertTrue(no_decay.values.all() == 1)  # without decay all weights are 1.0

        # negative decay sets 0 weights to approximately decay * length part of data set
        self.assertTrue(neg_decay[neg_decay == 0].shape[0] == 3)

        # in positive decay, weights decrease with increasing label time
        self.assertTrue(pos_decay.iloc[0] == pos_decay.max())
        self.assertTrue(pos_decay.iloc[-2] >= pos_decay.iloc[-1])

    def test_value_error_raise(self):
        """
        Test seq_bootstrap and ind_matrix functions for raising ValueError on nan values
        """

        with self.assertRaises(AssertionError):
            get_weights_by_return(self.meta_labeled_events, self.data['close'])

        with self.assertRaises(AssertionError):
            get_weights_by_time_decay(self.meta_labeled_events, self.data['close'], decay=0.5)
