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
from mlfinlab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix, get_ind_mat_average_uniqueness


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
                                                 timedelta=pd.Timedelta('2D'))

        self.data['side'] = 1
        self.meta_labeled_events = get_events(close=self.data['close'],
                                              t_events=cusum_events,
                                              pt_sl=[4, 4],
                                              target=daily_vol,
                                              min_ret=0.005,
                                              num_threads=3,
                                              vertical_barrier_times=vertical_barriers,
                                              side_prediction=self.data['side'])

    def test_num_concurrent_events(self):
        """
        Assert that number of concurent events have are available for all labels and equal to particular values
        """

        num_conc_events = num_concurrent_events(self.data['close'].index, self.meta_labeled_events['t1'],
                                                self.meta_labeled_events.index)
        # assert for each label we have concurrency value
        self.assertTrue(num_conc_events[self.meta_labeled_events.index].shape[0] == self.meta_labeled_events.shape[0])
        self.assertTrue(num_conc_events.value_counts()[0] == 186)
        self.assertTrue(num_conc_events.value_counts()[1] == 505)
        self.assertTrue(num_conc_events.value_counts()[2] == 92)

    def test_get_av_uniqueness(self):
        """
        Assert that average event uniqueness is available for all labels and equals to particular values
        """

        av_un = get_av_uniqueness_from_tripple_barrier(self.meta_labeled_events, self.data['close'], num_threads=4)
        # assert for each label we have uniqueness value
        self.assertTrue(av_un.shape[0] == self.meta_labeled_events.shape[0])
        self.assertTrue(av_un['tW'].iloc[0] == 1)
        self.assertTrue(av_un['tW'].iloc[4] == 0.5)
        self.assertTrue(av_un['tW'].iloc[6] == 0.85)
        self.assertTrue(bool(pd.isnull(av_un['tW'].iloc[-1])) is True)

    def test_seq_bootstrap(self):
        """
        Test sequential bootstrapping length, indicator matrix length and NaN checks
        """
        # Test get_ind_matrix function
        try:
            get_ind_matrix(self.meta_labeled_events)  # bar index contains NaN, which must be handled
        except ValueError:
            non_nan_meta_labels = self.meta_labeled_events.dropna()
            ind_mat = get_ind_matrix(non_nan_meta_labels)
        self.assertTrue(ind_mat.shape == (13, 7))

        self.assertTrue((ind_mat[2].values == np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])).all())

        bootstrapped_samples = seq_bootstrap(ind_mat, compare=False)
        bootstrapped_samples_1000 = seq_bootstrap(ind_mat, compare=True, sample_length=100)
        self.assertTrue(len(bootstrapped_samples) == non_nan_meta_labels.shape[0])
        self.assertTrue(len(bootstrapped_samples_1000) == 100)

        # check average uniqueness value
        sequential_unq = get_ind_mat_average_uniqueness(ind_mat[bootstrapped_samples_1000].values)
        sequential_unq_mean = sequential_unq[sequential_unq > 0].mean()

        self.assertTrue(abs(0.05 - sequential_unq_mean) <= 1e-2)  # sequential uniqueness should be higher

