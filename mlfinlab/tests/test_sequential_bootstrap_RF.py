"""
Test Random Forest with sequential bootstrapping regarding chapter 6: Ensemble Methods.
"""

import os
import unittest

import pandas as pd

from mlfinlab.ensemble.SeqBootstrapRandomForest import SeqBootstrapRandomForest
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import add_vertical_barrier, get_events, get_bins
from mlfinlab.util.utils import get_daily_vol


class TestChapter6(unittest.TestCase):
    """
    Test Random Forest modified to replace standard bootstrapping with sequential bootstrapping.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

    def test_random_forest_with_seq_bootstraping(self):
        """
        Assert that sequential bootstrapping in Random Forest modification works.
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

        # Training set
        X = self.data[["close", "cum_vol"]].loc[triple_labels.index]
        y = triple_labels["bin"]

        clf = SeqBootstrapRandomForest(triple_barrier_events=triple_barrier_events)
        clf.fit(X, y)
        pred_target = clf.predict(X)

        # Assert data type
        self.assertTrue(pred_target.dtype == int)

        # Assert shape
        self.assertTrue(triple_labels.shape[0] == pred_target.shape[0])
