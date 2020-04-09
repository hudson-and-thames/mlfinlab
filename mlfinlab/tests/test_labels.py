"""
Test various functions regarding chapter 3: Labels.
"""

import os
import unittest

import numpy as np
import pandas as pd

from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import add_vertical_barrier, get_events, get_bins, drop_labels
from mlfinlab.util.volatility import get_daily_vol


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

        # test localized datetimes
        self.data.index = self.data.index.tz_localize(tz='UTC')
        daily_vol_tz = get_daily_vol(close=self.data['close'], lookback=100)
        self.assertTrue((daily_vol.dropna().values == daily_vol_tz.dropna().values).all())

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

        # Test how labelling works with tz-aware timestamp
        self.data.index = self.data.index.tz_localize(tz='US/Eastern')
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
        self.assertTrue(triple_barrier_events.shape == (8, 4))  # Assert shape

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
        self.assertTrue(meta_labeled_events.shape == (8, 5))

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
        self.assertTrue(no_vertical_events.shape == (8, 4))

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

    def test_pt_sl_levels_triple_barrier_events(self):
        """
        Previously a bug was introduced by not multiplying the target by the profit taking / stop loss multiple. This
        meant that the get_bins function would not return the correct label. Example: if take profit was set to 1000,
        it would ignore this multiple and use only the target value. This meant that if we set a very large pt value
        (so high that it would never be hit before the vertical barrier is reached), it would ignore the multiple and
        only use the target value (it would signal that price reached the pt barrier). This meant that vertical barriers
        were incorrectly labeled.

        This also meant that irrespective of the pt_sl levels set, the labels would always be the same.
        """

        target = get_daily_vol(close=self.data['close'], lookback=100)
        cusum_events = cusum_filter(self.data['close'], threshold=0.02)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'], num_days=1)

        # --------------------------------------------------------------------------------------------------------
        # Assert that the vertical barrier would be reached for all positions due to the high pt level.
        # All labels should be 0. Check the 'bin' column
        pt_sl = [1000, 1000]
        triple_barrier_events_ptsl = get_events(close=self.data['close'],
                                                t_events=cusum_events,
                                                pt_sl=pt_sl,
                                                target=target,
                                                min_ret=0.005,
                                                num_threads=3,
                                                vertical_barrier_times=vertical_barriers,
                                                side_prediction=None)

        triple_labels_ptsl_large = get_bins(triple_barrier_events_ptsl, self.data['close'])
        labels_large = triple_labels_ptsl_large['bin']
        label_count = triple_labels_ptsl_large['bin'].sum()
        self.assertTrue(label_count == 0)

        # --------------------------------------------------------------------------------------------------------
        # Assert that the vertical barriers are never reached for very small multiples
        triple_barrier_events_ptsl = get_events(close=self.data['close'],
                                                t_events=cusum_events,
                                                pt_sl=[0.00000001, 0.00000001],
                                                target=target,
                                                min_ret=0.005,
                                                num_threads=3,
                                                vertical_barrier_times=vertical_barriers,
                                                side_prediction=None)

        triple_labels_ptsl_small = get_bins(triple_barrier_events_ptsl, self.data['close'])
        labels_small = triple_labels_ptsl_small['bin']
        label_count = (triple_labels_ptsl_small['bin'] == 0).sum()
        self.assertTrue(label_count == 0)

        # --------------------------------------------------------------------------------------------------------
        # TP too large and tight stop loss: expected all values less than 1
        triple_barrier_events_ptsl = get_events(close=self.data['close'],
                                                t_events=cusum_events,
                                                pt_sl=[10000, 0.00000001],
                                                target=target,
                                                min_ret=0.005,
                                                num_threads=3,
                                                vertical_barrier_times=vertical_barriers,
                                                side_prediction=None)

        labels_no_ones = get_bins(triple_barrier_events_ptsl, self.data['close'])['bin']
        self.assertTrue(np.all(labels_no_ones < 1))

        # --------------------------------------------------------------------------------------------------------
        # Test that the bins are in-fact different. (Previously they would be the same)
        self.assertTrue(np.all(labels_small[0:5] != labels_large[0:5]))

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
