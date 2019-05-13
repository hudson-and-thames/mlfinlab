"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier
from mlfinlab.util.utils import get_daily_vol
from mlfinlab.sampling.concurrent import get_av_uniqueness_from_tripple_barrier, num_concurrent_events
from mlfinlab.sampling.bootstrapping import seq_bootstrap


class TestSampling(unittest.TestCase):
    """
    Test Triple barrier, meta-labeling, dropping rare labels, and daily volatility.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

    def test_num_concurrent_events(self):
        """
        Assert that the different version of triple barrier labeling match our expected output.
        Assert that trgts are the same for all 3 methods.
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                 timedelta=pd.Timedelta('2D'))

        self.data['side'] = 1
        meta_labeled_events = get_events(close=self.data['close'],
                                         t_events=cusum_events,
                                         pt_sl=[4, 4],
                                         target=daily_vol,
                                         min_ret=0.005,
                                         num_threads=3,
                                         vertical_barrier_times=vertical_barriers,
                                         side_prediction=self.data['side'])

        num_conc_events = num_concurrent_events(self.data['close'].index, meta_labeled_events['t1'],
                                                meta_labeled_events.index)
        # assert for each label we have concurrency value
        self.assertTrue(num_conc_events[meta_labeled_events.index].shape[0] == meta_labeled_events.shape[0])
        self.assertTrue(num_conc_events.value_counts()[0] == 186)
        self.assertTrue(num_conc_events.value_counts()[1] == 505)
        self.assertTrue(num_conc_events.value_counts()[2] == 92)

    def test_get_av_uniqueness(self):
        """
        Assert that the different version of triple barrier labeling match our expected output.
        Assert that trgts are the same for all 3 methods.
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                 timedelta=pd.Timedelta('2D'))

        self.data['side'] = 1
        meta_labeled_events = get_events(close=self.data['close'],
                                         t_events=cusum_events,
                                         pt_sl=[4, 4],
                                         target=daily_vol,
                                         min_ret=0.005,
                                         num_threads=3,
                                         vertical_barrier_times=vertical_barriers,
                                         side_prediction=self.data['side'])

        av_un = get_av_uniqueness_from_tripple_barrier(meta_labeled_events, self.data['close'], num_threads=4)
        # assert for each label we have uniqueness value
        self.assertTrue(av_un.shape[0] == meta_labeled_events.shape[0])
        self.assertTrue(av_un['tW'].iloc[0] == 1)
        self.assertTrue(av_un['tW'].iloc[4] == 0.5)
        self.assertTrue(av_un['tW'].iloc[6] == 0.85)
        self.assertTrue(bool(pd.isnull(av_un['tW'].iloc[-1])) is True)



