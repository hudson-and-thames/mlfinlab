"""
Test various functions regarding chapter 3: Labels.
"""

import os
import unittest

import numpy as np
import pandas as pd

from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import add_vertical_barrier, get_events, get_bins, drop_labels
from mlfinlab.util.utils import get_daily_vol


class TestChapter3(unittest.TestCase):
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

    def test_daily_volatility(self):
        """
        Daily vol as implemented here matches the code in the book.
        Although I have reservations, example: no minimum value is set in the EWM.
        Thus it returns values for volatility before there are even enough data points.
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)

        # Last value in the set is still the same
        self.assertTrue(daily_vol[-1] == 0.008968238932170641)

        # Size matches
        self.assertTrue(daily_vol.shape[0] == 960)

    def test_vertical_barriers(self):
        """
        Assert that the vertical barrier returns the timestamp x amount of days after the event.
        """
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)

        # Compute vertical barrier
        for days in [1, 2, 3, 4, 5]:
            vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=days)

            # For each row assert the time delta is correct
            for start_date, end_date in vertical_barriers.iteritems():
                self.assertTrue((end_date - start_date).days >= 1)

        # Check hourly barriers
        for hours in [1, 2, 3, 4, 5]:
            vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_hours=hours)

            # For each row assert the time delta is correct
            for start_date, end_date in vertical_barriers.iteritems():
                self.assertTrue((end_date - start_date).seconds >= 3600)

        # Check minute barriers
        for minutes in [1, 2, 3, 4, 5]:
            vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                     num_minutes=minutes)

            # For each row assert the time delta is correct
            for start_date, end_date in vertical_barriers.iteritems():
                self.assertTrue((end_date - start_date).seconds >= 60)

        # Check seconds barriers
        for seconds in [1, 2, 3, 4, 5]:
            vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                     num_seconds=seconds)

            # For each row assert the time delta is correct
            for start_date, end_date in vertical_barriers.iteritems():
                self.assertTrue((end_date - start_date).seconds >= 1)

    def test_triple_barrier_events(self):
        """
        Assert that the different version of triple barrier labeling match our expected output.
        Assert that trgts are the same for all 3 methods.
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=1)

        # No meta-labeling
        triple_barrier_events = get_events(close=self.data['close'],
                                           t_events=cusum_events,
                                           pt_sl=[1, 1],
                                           target=daily_vol,
                                           min_ret=0.005,
                                           num_threads=3,
                                           vertical_barrier_times=vertical_barriers,
                                           side_prediction=None)

        # Test that the events are the same as expected (naive test)
        self.assertTrue(triple_barrier_events.shape == (8, 2))  # Assert shape

        # Assert that targets match expectations
        self.assertTrue(triple_barrier_events.iloc[0, 1] == 0.010166261175903357)
        self.assertTrue(triple_barrier_events.iloc[-1, 1] == 0.006455887663302871)

        # Assert start of triple barrier event aligns with cusum_filter
        self.assertTrue(np.all(triple_barrier_events.index == cusum_events[1:]))

        # ----------------------
        # With meta-labeling
        self.data['side'] = 1
        meta_labeled_events = get_events(close=self.data['close'],
                                         t_events=cusum_events,
                                         pt_sl=[1, 1],
                                         target=daily_vol,
                                         min_ret=0.005,
                                         num_threads=3,
                                         vertical_barrier_times=vertical_barriers,
                                         side_prediction=self.data['side'])

        # Assert that the two different events are the the same as they are generated using same data
        self.assertTrue(np.all(meta_labeled_events['t1'] == triple_barrier_events['t1']))
        self.assertTrue(np.all(meta_labeled_events['trgt'] == triple_barrier_events['trgt']))

        # Assert shape
        self.assertTrue(meta_labeled_events.shape == (8, 3))

        # ----------------------
        # No vertical barriers
        no_vertical_events = get_events(close=self.data['close'],
                                        t_events=cusum_events,
                                        pt_sl=[1, 1],
                                        target=daily_vol,
                                        min_ret=0.005,
                                        num_threads=3,
                                        vertical_barrier_times=False,
                                        side_prediction=None)

        # Assert targets match other events trgts
        self.assertTrue(np.all(triple_barrier_events['trgt'] == no_vertical_events['trgt']))
        self.assertTrue(no_vertical_events.shape == (8, 2))

        # Previously the vertical barrier was touched twice, assert that those events aren't included here
        self.assertTrue((no_vertical_events['t1'] != triple_barrier_events['t1']).sum() == 2)

    def test_triple_barrier_labeling(self):
        """
        Assert that meta labeling as well as standard labeling works. Also check that if a vertical barrier is
        reached, then a 0 class label is assigned (in the case of standard labeling).
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=1)

        # ----------------------
        # Assert 0 labels are generated if vertical barrier hit
        triple_barrier_events = get_events(close=self.data['close'],
                                           t_events=cusum_events,
                                           pt_sl=[1, 1],
                                           target=daily_vol,
                                           min_ret=0.005,
                                           num_threads=3,
                                           vertical_barrier_times=vertical_barriers,
                                           side_prediction=None)

        triple_labels = get_bins(triple_barrier_events, self.data['close'])
        self.assertTrue(np.all(triple_labels[np.abs(triple_labels['ret']) < triple_labels['trgt']]['bin'] == 0))

        # ----------------------
        # Assert meta labeling works
        self.data['side'] = 1
        triple_barrier_events = get_events(close=self.data['close'],
                                           t_events=cusum_events,
                                           pt_sl=[1, 1],
                                           target=daily_vol,
                                           min_ret=0.005,
                                           num_threads=3,
                                           vertical_barrier_times=vertical_barriers,
                                           side_prediction=self.data['side'])

        triple_labels = get_bins(triple_barrier_events, self.data['close'])

        # Label 1 if made money, else 0
        condition1 = triple_labels['ret'] > 0
        condition2 = triple_labels['ret'].abs() > triple_labels['trgt']
        self.assertTrue(((condition1 & condition2) == triple_labels['bin']).all())

        # Assert shape
        self.assertTrue(triple_labels.shape == (8, 4))

    def test_drop_labels(self):
        """
        Assert that drop_labels removes rare class labels.
        """
        daily_vol = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=1)
        triple_barrier_events = get_events(close=self.data['close'],
                                           t_events=cusum_events,
                                           pt_sl=[1, 1],
                                           target=daily_vol,
                                           min_ret=0.005,
                                           num_threads=3,
                                           vertical_barrier_times=vertical_barriers,
                                           side_prediction=None)
        triple_labels = get_bins(triple_barrier_events, self.data['close'])

        # Drop the 2 zero labels in the set since they are "rare"
        new_labels = drop_labels(events=triple_labels, min_pct=0.30)
        self.assertTrue(0 not in set(new_labels['bin']))

        # Assert that threshold works
        new_labels = drop_labels(events=triple_labels, min_pct=0.20)
        self.assertTrue(0 in set(new_labels['bin']))
