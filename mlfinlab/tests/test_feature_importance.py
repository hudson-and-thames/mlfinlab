"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import os
import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from mlfinlab.util.utils import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.ensemble.sb_bagging_classifier import SequentiallyBootstrappedBaggingClassifier
from mlfinlab.feature_importance.importance import feature_importance_mean_imp_reduction, \
    feature_importance_mean_decrease_accuracy, feature_importance_sfi
from mlfinlab.feature_importance.orthogonal import feature_pca_analysis, get_orthogonal_features
from mlfinlab.cross_validation.cross_validation import PurgedKFold


# pylint: disable=invalid-name


class TestSequentiallyBootstrappedBagging(unittest.TestCase):
    """
    Test SequentiallyBootstrapped Bagging classifiers
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data and get triple barrier events, generate features
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

        # Compute moving averages
        fast_window = 20
        slow_window = 50

        self.data['fast_mavg'] = self.data['close'].rolling(window=fast_window, min_periods=fast_window,
                                                            center=False).mean()
        self.data['slow_mavg'] = self.data['close'].rolling(window=slow_window, min_periods=slow_window,
                                                            center=False).mean()

        # Compute sides
        self.data['side'] = np.nan

        long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
        short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
        self.data.loc[long_signals, 'side'] = 1
        self.data.loc[short_signals, 'side'] = -1

        # Remove Look ahead bias by lagging the signal
        self.data['side'] = self.data['side'].shift(1)

        daily_vol = get_daily_vol(close=self.data['close'], lookback=50)
        cusum_events = cusum_filter(self.data['close'], threshold=0.001)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                 num_days=2)
        self.meta_labeled_events = get_events(close=self.data['close'],
                                              t_events=cusum_events,
                                              pt_sl=[4, 4],
                                              target=daily_vol,
                                              min_ret=0.005,
                                              num_threads=3,
                                              vertical_barrier_times=vertical_barriers,
                                              side_prediction=self.data['side'])

        self.meta_labeled_events.dropna(inplace=True)
        labels = get_bins(self.meta_labeled_events, self.data['close'])

        # Feature generation
        features = []
        X = self.data.copy()
        X['log_ret'] = X.close.apply(np.log).diff()
        for win in [2, 5, 10, 20, 25]:
            X['momentum_{}'.format(win)] = X.close / X.close.rolling(window=win).mean() - 1
            X['std_{}'.format(win)] = X.log_ret.rolling(window=win).std()
            X['pct_change_{}'.format(win)] = X.close.pct_change(win)
            X['diff_{}'.format(win)] = X.close.diff(win)

            for f in ['momentum', 'std', 'pct_change', 'diff']:
                features.append('{}_{}'.format(f, win))

        # Train/test generation
        X.dropna(inplace=True)
        X = X.loc[self.meta_labeled_events.index, :]  # Take only filtered events
        labels = labels.loc[X.index, :]  # Sync X and y
        self.meta_labeled_events = self.meta_labeled_events.loc[X.index, :]  # Sync X and meta_labeled_events

        self.X_train, self.y_train_clf, = X.iloc[:300][features], labels.iloc[:300].bin
        self.X_test, self.y_test_clf = X.iloc[300:][features], labels.iloc[300:].bin

    def test_sb_classifier(self):
        """
        Test Sequentially Bootstrapped Bagging Classifier
        """

        # Init classifiers
        clf = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                     class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=1.0, n_estimators=100,
                                                           triple_barrier_events=self.meta_labeled_events,
                                                           price_bars=self.data, oob_score=False, random_state=1)

        sb_clf.fit(self.X_train, self.y_train_clf)

        mdi_feat_imp = feature_importance_mean_imp_reduction(sb_clf, self.X_train.columns)
        feature_pca_analysis(self.X_train, mdi_feat_imp)

        triple_barrier_events = self.meta_labeled_events.loc[self.X_train.index, :]
        imp = feature_importance_mean_decrease_accuracy(sb_clf, self.X_train, self.y_train_clf, triple_barrier_events, n_splits=3)
        cv_gen = PurgedKFold(n_splits=4, info_sets=triple_barrier_events.t1)
        imp = feature_importance_sfi(sb_clf, self.X_train[self.X_train.columns[:5]], self.y_train_clf, cv_gen=cv_gen)
        print(imp)
