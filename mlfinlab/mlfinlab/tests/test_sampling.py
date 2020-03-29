"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import unittest

import numpy as np
import pandas as pd

from mlfinlab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix, get_ind_mat_average_uniqueness, \
    _bootstrap_loop_run, get_ind_mat_label_uniqueness  # pylint: disable=protected-access
from mlfinlab.sampling.concurrent import get_av_uniqueness_from_triple_barrier, num_concurrent_events


def book_ind_mat_implementation(bar_index, label_endtime):
    """
    Book implementation of get_ind_matrix function
    """
    ind_mat = pd.DataFrame(0, index=bar_index, columns=range(label_endtime.shape[0]))
    for i, (start, end) in enumerate(label_endtime.iteritems()):
        ind_mat.loc[start:end, i] = 1.
    return ind_mat


class TestSampling(unittest.TestCase):
    """
    Test Triple barrier, meta-labeling, dropping rare labels, and daily volatility.
    """

    def setUp(self):
        """
        Set samples_info_sets (t1), price bars
        """
        self.price_bars = pd.Series(index=pd.date_range(start="1/1/2018", end='1/8/2018', freq='H'))
        self.samples_info_sets = pd.DataFrame(index=self.price_bars.index[[1, 2, 5, 7, 10, 11, 12, 20]])
        self.samples_info_sets['t1'] = self.samples_info_sets.index + pd.Timedelta('2H')

    def test_num_concurrent_events(self):
        """
        Assert that number of concurent events have are available for all labels and equal to particular values
        """

        num_conc_events = num_concurrent_events(self.price_bars.index, self.samples_info_sets['t1'],
                                                self.samples_info_sets.index)
        # Assert for each label we have concurrency value
        self.assertTrue(num_conc_events[self.samples_info_sets.index].shape[0] == self.samples_info_sets.shape[0])
        self.assertTrue(num_conc_events.value_counts()[0] == 5)  # Hours between 14 and 20 label
        self.assertTrue(num_conc_events.value_counts()[1] == 11)
        self.assertTrue(num_conc_events.value_counts()[2] == 5)
        self.assertTrue(num_conc_events.value_counts()[3] == 1)  # Label # 10

    def test_get_av_uniqueness(self):
        """
        Assert that average event uniqueness is available for all labels and equals to particular values
        """

        av_un = get_av_uniqueness_from_triple_barrier(self.samples_info_sets, self.price_bars, num_threads=4)
        # Assert for each label we have uniqueness value
        self.assertTrue(av_un.shape[0] == self.samples_info_sets.shape[0])
        self.assertAlmostEqual(av_un['tW'].iloc[0], 0.66, delta=1e-2)
        self.assertAlmostEqual(av_un['tW'].iloc[2], 0.83, delta=1e-2)
        self.assertAlmostEqual(av_un['tW'].iloc[5], 0.44, delta=1e-2)
        self.assertAlmostEqual(av_un['tW'].iloc[-1], 1.0, delta=1e-2)

    def test_seq_bootstrap(self):
        """
        Test sequential bootstrapping length, indicator matrix length and NaN checks
        """

        ind_mat = get_ind_matrix(self.samples_info_sets.t1, self.price_bars)

        label_endtime = self.samples_info_sets.t1
        trimmed_price_bars_index = self.price_bars[(self.price_bars.index >= self.samples_info_sets.index.min()) &
                                                   (self.price_bars.index <= self.samples_info_sets.t1.max())].index
        bar_index = list(self.samples_info_sets.index)  # Generate index for indicator matrix from t1 and index
        bar_index.extend(self.samples_info_sets.t1)
        bar_index.extend(trimmed_price_bars_index)
        bar_index = sorted(list(set(bar_index)))  # Drop duplicates and sort
        ind_mat_book_implementation = book_ind_mat_implementation(bar_index, label_endtime)

        self.assertTrue(bool((ind_mat_book_implementation.values == ind_mat).all()) is True)
        # Indicator matrix shape should be (unique(meta_label_index+t1+price_bars_index), t1)
        self.assertTrue(ind_mat.shape == (22, 8))

        # Check indicator matrix values for specific labels
        self.assertTrue(bool((ind_mat[:3, 0] == np.ones(3)).all()) is True)
        self.assertTrue(bool((ind_mat[1:4, 1] == np.ones(3)).all()) is True)
        self.assertTrue(bool((ind_mat[4:7, 2] == np.ones(3)).all()) is True)
        self.assertTrue(bool((ind_mat[14:, 6] == np.zeros(8)).all()) is True)

        bootstrapped_samples = seq_bootstrap(ind_mat, compare=False, verbose=True, warmup_samples=None)
        bootstrapped_samples_1000 = seq_bootstrap(ind_mat, compare=True, sample_length=100)
        self.assertTrue(len(bootstrapped_samples) == self.samples_info_sets.shape[0])
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
            sequential_unq = get_ind_mat_average_uniqueness(ind_mat[:, bootstrapped_samples])

            standard_unq_array[i] = random_unq
            seq_unq_array[i] = sequential_unq

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
        self.assertTrue(abs(labels_av_uniqueness - 0.8571) <= 1e-4)  # Test matrix av.uniqueness

    def test_get_ind_mat_uniqueness(self):
        """
        Tests get_ind_mat_uniqueness function using indicator matrix from the book example
        """

        ind_mat = pd.DataFrame(index=range(0, 6), columns=range(0, 3))
        ind_mat.loc[:, 0] = [1, 1, 1, 0, 0, 0]
        ind_mat.loc[:, 1] = [0, 0, 1, 1, 0, 0]
        ind_mat.loc[:, 2] = [0, 0, 0, 0, 1, 1]
        ind_mat = ind_mat.values

        labels_av_uniqueness = get_ind_mat_label_uniqueness(ind_mat)
        first_sample_unq = labels_av_uniqueness[0]
        second_sample_unq = labels_av_uniqueness[1]
        third_sample_unq = labels_av_uniqueness[2]

        self.assertTrue(abs(first_sample_unq[first_sample_unq > 0].mean() - 0.8333) <= 1e-4)
        self.assertTrue(abs(second_sample_unq[second_sample_unq > 0].mean() - 0.75) <= 1e-4)
        self.assertTrue(abs(third_sample_unq[third_sample_unq > 0].mean() - 1.0) <= 1e-4)
        # Test matrix av.uniqueness
        self.assertTrue(abs(labels_av_uniqueness[labels_av_uniqueness > 0].mean() - 0.8571) <= 1e-4)

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
        nan_samples_info_sets = self.samples_info_sets.copy()
        nan_samples_info_sets.loc[pd.Timestamp(2019, 1, 1), 't1'] = None
        with self.assertRaises(ValueError):
            get_ind_matrix(nan_samples_info_sets.t1, self.price_bars)
