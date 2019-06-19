"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import os
import unittest

import numpy as np
import pandas as pd

from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier
from mlfinlab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix, get_ind_mat_average_uniqueness, \
    _bootstrap_loop_run  # pylint: disable=protected-access
from mlfinlab.sampling.concurrent import get_av_uniqueness_from_tripple_barrier, num_concurrent_events
from mlfinlab.util.utils import get_daily_vol


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

    def test_num_concurrent_events(self):
        """
        Assert that number of concurent events have are available for all labels and equal to particular values
        """

        num_conc_events = num_concurrent_events(self.data['close'].index, self.meta_labeled_events['t1'],
                                                self.meta_labeled_events.index)
        # Assert for each label we have concurrency value
        self.assertTrue(num_conc_events[self.meta_labeled_events.index].shape[0] == self.meta_labeled_events.shape[0])
        self.assertTrue(num_conc_events.value_counts()[0] == 186)
        self.assertTrue(num_conc_events.value_counts()[1] == 505)
        self.assertTrue(num_conc_events.value_counts()[2] == 92)

    def test_get_av_uniqueness(self):
        """
        Assert that average event uniqueness is available for all labels and equals to particular values
        """

        av_un = get_av_uniqueness_from_tripple_barrier(self.meta_labeled_events, self.data['close'], num_threads=4)
        # Assert for each label we have uniqueness value
        self.assertTrue(av_un.shape[0] == self.meta_labeled_events.shape[0])
        self.assertTrue(av_un['tW'].iloc[0] == 1)
        self.assertTrue(av_un['tW'].iloc[4] == 0.5)
        self.assertTrue(av_un['tW'].iloc[6] == 0.85)
        self.assertTrue(bool(pd.isnull(av_un['tW'].iloc[-1])) is True)

    def test_seq_bootstrap(self):
        """
        Test sequential bootstrapping length, indicator matrix length and NaN checks
        """

        non_nan_meta_labels = self.meta_labeled_events.dropna()
        ind_mat = get_ind_matrix(non_nan_meta_labels)
        self.assertTrue(ind_mat.shape == (13, 7))  # Indicator matrix shape should be (meta_label_index+t1, t1)
        # Check indicator matrix values for specific labels
        self.assertTrue(bool((ind_mat[:, 0] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all()) is True)
        self.assertTrue(bool((ind_mat[:, 2] == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]).all()) is True)
        self.assertTrue(bool((ind_mat[:, 4] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]).all()) is True)
        self.assertTrue(bool((ind_mat[:, 6] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).all()) is True)

        bootstrapped_samples = seq_bootstrap(ind_mat, compare=False, verbose=True, warmup_samples=None)
        bootstrapped_samples_1000 = seq_bootstrap(ind_mat, compare=True, sample_length=100)
        self.assertTrue(len(bootstrapped_samples) == non_nan_meta_labels.shape[0])
        self.assertTrue(len(bootstrapped_samples_1000) == 100)

        # Test sequential bootstrapping on example from a book
        ind_mat = pd.DataFrame(index=range(0, 6), columns=range(0, 3))
        ind_mat.loc[:, 0] = [1, 1, 1, 0, 0, 0]
        ind_mat.loc[:, 1] = [0, 0, 1, 1, 0, 0]
        ind_mat.loc[:, 2] = [0, 0, 0, 0, 1, 1]
        ind_mat = ind_mat.values

        seq_bootstrap(ind_mat, sample_length=3, verbose=True, warmup_samples=[1])  # Show printed probabilities

        # Perform Monte-Carlo test
        standard_unq_array = np.zeros(1000) * np.nan
        seq_unq_array = np.zeros(1000) * np.nan
        for i in range(0, 1000):
            bootstrapped_samples = seq_bootstrap(ind_mat, sample_length=3)
            random_samples = np.random.choice(ind_mat.shape[1], size=3)

            random_unq = get_ind_mat_average_uniqueness(ind_mat[:, random_samples])
            random_unq_mean = random_unq[random_unq > 0].mean()

            sequential_unq = get_ind_mat_average_uniqueness(ind_mat[:, bootstrapped_samples])
            sequential_unq_mean = sequential_unq[sequential_unq > 0].mean()

            standard_unq_array[i] = random_unq_mean
            seq_unq_array[i] = sequential_unq_mean

        self.assertTrue(np.mean(seq_unq_array) >= np.mean(standard_unq_array))
        self.assertTrue(np.median(seq_unq_array) >= np.median(standard_unq_array))

    def test_get_ind_mat_av_uniqueness(self):
        """
        Tests get_ind_mat_average_uniqueness function using indicator matrix from the book example
        """

        ind_mat = pd.DataFrame(index=range(0, 6), columns=range(0, 3))
        ind_mat.loc[:, 0] = [1, 1, 1, 0, 0, 0]
        ind_mat.loc[:, 1] = [0, 0, 1, 1, 0, 0]
        ind_mat.loc[:, 2] = [0, 0, 0, 0, 1, 1]
        ind_mat = ind_mat.values

        labels_av_uniqueness = get_ind_mat_average_uniqueness(ind_mat)
        first_sample_unq = labels_av_uniqueness[0]
        second_sample_unq = labels_av_uniqueness[1]
        third_sample_unq = labels_av_uniqueness[2]

        self.assertTrue(abs(first_sample_unq[first_sample_unq > 0].mean() - 0.8333) <= 1e-4)  # First sample uniqueness
        self.assertTrue(abs(second_sample_unq[second_sample_unq > 0].mean() - 0.75) <= 1e-4)
        self.assertTrue(abs(third_sample_unq[third_sample_unq > 0].mean() - 1.0) <= 1e-4)
        self.assertTrue(
            abs(labels_av_uniqueness[labels_av_uniqueness > 0].mean() - 0.8571) <= 1e-4)  # Test matrix av.uniqueness

    def test_bootstrap_loop_run(self):
        """
        Test one loop iteration of Sequential Bootstrapping
        """
        ind_mat = pd.DataFrame(index=range(0, 6), columns=range(0, 3))
        ind_mat.loc[:, 0] = [1, 1, 1, 0, 0, 0]
        ind_mat.loc[:, 1] = [0, 0, 1, 1, 0, 0]
        ind_mat.loc[:, 2] = [0, 0, 0, 0, 1, 1]
        ind_mat = ind_mat.values

        prev_concurrency = np.zeros(ind_mat.shape[0])

        first_iteration = _bootstrap_loop_run(ind_mat, prev_concurrency)
        self.assertTrue((first_iteration == np.array([1.0, 1.0, 1.0])).all())  # First iteration should always yield 1

        prev_concurrency += ind_mat[:, 1]  # Repeat example from the book
        second_iteration = _bootstrap_loop_run(ind_mat, prev_concurrency)
        second_iteration_prob = second_iteration / second_iteration.sum()

        self.assertTrue(abs((second_iteration_prob - np.array([0.35714286, 0.21428571, 0.42857143])).sum()) <= 1e-8)

    def test_value_error_raise(self):
        """
        Test seq_bootstrap and ind_matrix functions for raising ValueError on nan values
        """

        with self.assertRaises(ValueError):
            get_ind_matrix(self.meta_labeled_events)
