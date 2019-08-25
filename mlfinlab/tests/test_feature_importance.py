"""
Test various functions regarding chapter 8: MDI, MDA, SFI importance.
"""

import os
import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from mlfinlab.util.utils import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from mlfinlab.feature_importance.importance import (feature_importance_mean_imp_reduction,
                                                    feature_importance_mean_decrease_accuracy, feature_importance_sfi,
                                                    plot_feature_importance)
from mlfinlab.feature_importance.orthogonal import feature_pca_analysis, get_orthogonal_features
from mlfinlab.cross_validation.cross_validation import PurgedKFold, ml_cross_val_score


# pylint: disable=invalid-name


class TestFeatureImportance(unittest.TestCase):
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

    def test_orthogonal_features(self):
        """
        Test orthogonal features: PCA features, importance vs PCA importance analysis
        """

        # Init classifiers
        clf = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                     class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=1.0, n_estimators=100,
                                                           triple_barrier_events=self.meta_labeled_events,
                                                           price_bars=self.data, oob_score=False, random_state=1)

        pca_features = get_orthogonal_features(self.X_train)

        # PCA features should have mean of 0
        self.assertAlmostEqual(np.mean(pca_features[:, 2]), 0, delta=1e-7)
        self.assertAlmostEqual(np.mean(pca_features[:, 5]), 0, delta=1e-7)
        self.assertAlmostEqual(np.mean(pca_features[:, 6]), 0, delta=1e-7)

        # Check particular PCA values std
        self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.862, delta=1e-3)
        self.assertAlmostEqual(np.std(pca_features[:, 3]), 0.987, delta=1e-3)
        self.assertAlmostEqual(np.std(pca_features[:, 4]), 0.954, delta=1e-3)

        sb_clf.fit(self.X_train, self.y_train_clf)
        mdi_feat_imp = feature_importance_mean_imp_reduction(sb_clf, self.X_train.columns)
        pca_corr_res = feature_pca_analysis(self.X_train, mdi_feat_imp)

        # Check correlation metrics results
        self.assertAlmostEqual(pca_corr_res['Pearson'][0], -0.194, delta=1e-3)
        self.assertAlmostEqual(pca_corr_res['Spearman'][0], -0.178, delta=1e-3)
        self.assertAlmostEqual(pca_corr_res['Kendall'][0], -0.115, delta=1e-3)
        self.assertAlmostEqual(pca_corr_res['Weighted_Kendall_Rank'][0], -0.078, delta=1e-3)

    def test_feature_importance(self):
        """
        Test features importance: MDI, MDA, SFI and plot function
        """
        sb_clf, cv_gen = self._prepare_clf_data_set(oob_score=False)

        # MDI feature importance
        mdi_feat_imp = feature_importance_mean_imp_reduction(sb_clf, self.X_train.columns)

        # MDA feature importance
        mda_feat_imp_log_loss = feature_importance_mean_decrease_accuracy(sb_clf, self.X_train, self.y_train_clf,
                                                                          cv_gen,
                                                                          sample_weight=np.ones(
                                                                              (self.X_train.shape[0],)))
        mda_feat_imp_accuracy = feature_importance_mean_decrease_accuracy(sb_clf, self.X_train, self.y_train_clf,
                                                                          cv_gen, scoring='accuracy')
        # SFI feature importance
        sfi_feat_imp_log_loss = feature_importance_sfi(sb_clf, self.X_train[self.X_train.columns[:5]], self.y_train_clf,
                                                       cv_gen=cv_gen, sample_weight=np.ones((self.X_train.shape[0],)))
        sfi_feat_imp_accuracy = feature_importance_sfi(sb_clf, self.X_train[self.X_train.columns[:5]], self.y_train_clf,
                                                       cv_gen=cv_gen, scoring='accuracy')

        # MDI assertions
        self.assertTrue(mdi_feat_imp['mean'].sum() == 1)
        self.assertAlmostEqual(mdi_feat_imp.loc['momentum_2', 'mean'], 0.0434, delta=1e-3)
        self.assertAlmostEqual(mdi_feat_imp.loc['momentum_2', 'std'], 0.002779, delta=1e-3)
        self.assertAlmostEqual(mdi_feat_imp.loc['pct_change_5', 'mean'], 0.0434, delta=1e-3)
        self.assertAlmostEqual(mdi_feat_imp.loc['pct_change_5', 'std'], 0.00292, delta=1e-3)
        self.assertAlmostEqual(mdi_feat_imp.loc['std_20', 'mean'], 0.08421, delta=1e-3)

        # MDA(log_loss) assertions
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['diff_20', 'mean'], -0.026309, delta=1e-2)
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['diff_20', 'std'], 0.01824, delta=1e-2)

        # MDA(accuracy) assertions
        self.assertAlmostEqual(mda_feat_imp_accuracy.loc['diff_20', 'std'], 0.0485, delta=0.7)

        # SFI(log_loss) assertions
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['momentum_2', 'mean'], -2.879, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['momentum_2', 'std'], 0.66422, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['diff_2', 'mean'], -2.0558, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['diff_2', 'std'], 0.434, delta=1e-2)

        # SFI(accuracy) assertions
        self.assertAlmostEqual(sfi_feat_imp_accuracy.loc['momentum_2', 'mean'], 0.51, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_accuracy.loc['momentum_2', 'std'], 0.05361, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_accuracy.loc['diff_2', 'mean'], 0.533, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_accuracy.loc['diff_2', 'std'], 0.04027, delta=1e-2)

    def test_plot_feature_importance(self):
        """
        Test plot_feature_importance function
        """

        sb_clf, cv_gen = self._prepare_clf_data_set(oob_score=True)
        oos_score = ml_cross_val_score(sb_clf, self.X_train, self.y_train_clf, cv_gen=cv_gen, sample_weight=None,
                                       scoring='accuracy').mean()

        mdi_feat_imp = feature_importance_mean_imp_reduction(sb_clf, self.X_train.columns)
        plot_feature_importance(mdi_feat_imp, oob_score=sb_clf.oob_score_, oos_score=oos_score)
        plot_feature_importance(mdi_feat_imp, oob_score=sb_clf.oob_score_, oos_score=oos_score,
                                savefig=True, output_path='test.png')

        os.remove('test.png')

    def _prepare_clf_data_set(self, oob_score):
        """
        Helper function for preparing data sets for feature importance

        :param oob_score: (bool): bool flag for oob_score in classifier
        """
        clf = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                     class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=1.0, n_estimators=100,
                                                           triple_barrier_events=self.meta_labeled_events,
                                                           price_bars=self.data, oob_score=oob_score, random_state=1)
        sb_clf.fit(self.X_train, self.y_train_clf)

        triple_barrier_events = self.meta_labeled_events.loc[self.X_train.index, :]
        cv_gen = PurgedKFold(n_splits=4, info_sets=triple_barrier_events.t1, random_state=1)
        return sb_clf, cv_gen

    def test_raise_value_error(self):
        """
        Test ValueError raise in MDA, SFI
        """
        sb_clf, cv_gen = self._prepare_clf_data_set(oob_score=False)

        with self.assertRaises(ValueError):
            feature_importance_mean_decrease_accuracy(sb_clf, self.X_train, self.y_train_clf,
                                                      cv_gen, sample_weight=np.ones((self.X_train.shape[0],)),
                                                      scoring='roc')
        with self.assertRaises(ValueError):
            feature_importance_sfi(sb_clf, self.X_train[self.X_train.columns[:5]], self.y_train_clf,
                                   cv_gen=cv_gen, sample_weight=np.ones((self.X_train.shape[0],)), scoring='roc')
