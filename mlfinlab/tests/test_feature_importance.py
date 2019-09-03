"""
Test various functions regarding chapter 8: MDI, MDA, SFI importance.
"""

import os
import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlfinlab.util.utils import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.sampling.bootstrapping import get_ind_mat_label_uniqueness, get_ind_matrix
from mlfinlab.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from mlfinlab.feature_importance.importance import (feature_importance_mean_decrease_impurity,
                                                    feature_importance_mean_decrease_accuracy, feature_importance_sfi,
                                                    plot_feature_importance)
from mlfinlab.feature_importance.orthogonal import feature_pca_analysis, get_orthogonal_features
from mlfinlab.cross_validation.cross_validation import PurgedKFold, ml_cross_val_score


# pylint: disable=invalid-name


def _generate_label_with_prob(x, prob, random_state=np.random.RandomState(1)):
    """
    Generates true label value with some probability(prob)
    """
    choice = random_state.choice([0, 1], p=[1 - prob, prob])
    if choice == 1:
        return x
    return int(not x)


def _get_synthetic_samples(ind_mat, good_samples_thresh, bad_samples_thresh):
    """
    Get samples with uniqueness either > good_samples_thresh or uniqueness < bad_samples_thresh
    """
    # Get mix of samples where some of them are extremely non-overlapping, the other one are highly overlapping
    i = 0
    unique_samples = []
    for label in get_ind_mat_label_uniqueness(ind_mat):
        if np.mean(label[label > 0]) > good_samples_thresh or np.mean(label[label > 0]) < bad_samples_thresh:
            unique_samples.append(i)
        i += 1
    return unique_samples


class TestFeatureImportance(unittest.TestCase):
    """
    Test Feature importance
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
        self.data['fast_mavg'] = self.data['close'].rolling(window=20, min_periods=20,
                                                            center=False).mean()
        self.data['slow_mavg'] = self.data['close'].rolling(window=50, min_periods=50,
                                                            center=False).mean()

        # Compute sides
        self.data['side'] = np.nan

        long_signals = self.data['fast_mavg'] >= self.data['slow_mavg']
        short_signals = self.data['fast_mavg'] < self.data['slow_mavg']
        self.data.loc[long_signals, 'side'] = 1
        self.data.loc[short_signals, 'side'] = -1

        # Remove Look ahead bias by lagging the signal
        self.data['side'] = self.data['side'].shift(1)

        daily_vol = get_daily_vol(close=self.data['close'], lookback=50) * 0.5
        cusum_events = cusum_filter(self.data['close'], threshold=0.005)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=self.data['close'],
                                                 num_hours=2)
        meta_labeled_events = get_events(close=self.data['close'],
                                         t_events=cusum_events,
                                         pt_sl=[1, 4],
                                         target=daily_vol,
                                         min_ret=5e-5,
                                         num_threads=3,
                                         vertical_barrier_times=vertical_barriers,
                                         side_prediction=self.data['side'])
        meta_labeled_events.dropna(inplace=True)
        labels = get_bins(meta_labeled_events, self.data['close'])

        # Generate data set which shows the power of SB Bagging vs Standard Bagging
        ind_mat = get_ind_matrix(meta_labeled_events.t1, self.data.close)

        unique_samples = _get_synthetic_samples(ind_mat, 0.5, 0.1)

        X = self.data.loc[labels.index,].iloc[unique_samples].dropna()  # get synthetic data set with drawn samples
        labels = labels.loc[X.index, :]
        X.loc[labels.index, 'y'] = labels.bin

        # Generate features (some of them are informative, others are just noise)
        for index, value in X.y.iteritems():
            X.loc[index, 'label_prob_0.6'] = _generate_label_with_prob(value, 0.6)
            X.loc[index, 'label_prob_0.5'] = _generate_label_with_prob(value, 0.5)
            X.loc[index, 'label_prob_0.3'] = _generate_label_with_prob(value, 0.3)
            X.loc[index, 'label_prob_0.2'] = _generate_label_with_prob(value, 0.2)
            X.loc[index, 'label_prob_0.1'] = _generate_label_with_prob(value, 0.1)

        features = ['label_prob_0.6', 'label_prob_0.2', 'label_prob_0.1']  # Two super-informative features
        for prob in [0.5, 0.3, 0.2, 0.1]:
            for window in [2, 5]:
                X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(
                    window=window).mean()
                features.append('label_prob_{}_sma_{}'.format(prob, window))
        X.dropna(inplace=True)
        y = X.pop('y')

        self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4,
                                                                                        random_state=1, shuffle=False)
        self.y_train_reg = (1 + self.y_train_clf)
        self.y_test_reg = (1 + self.y_test_clf)

        self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
        self.price_bars_trim = self.data[
            (self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

    def test_orthogonal_features(self):
        """
        Test orthogonal features: PCA features, importance vs PCA importance analysis
        """

        # Init classifiers
        clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                          class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_base, max_features=1.0, n_estimators=100,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=True,
                                                           random_state=1)

        pca_features = get_orthogonal_features(self.X_train)

        # PCA features should have mean of 0
        self.assertAlmostEqual(np.mean(pca_features[:, 2]), 0, delta=1e-7)
        self.assertAlmostEqual(np.mean(pca_features[:, 5]), 0, delta=1e-7)
        self.assertAlmostEqual(np.mean(pca_features[:, 6]), 0, delta=1e-7)

        # Check particular PCA values std
        self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.499, delta=0.2)
        self.assertAlmostEqual(np.std(pca_features[:, 3]), 1.047, delta=0.2)
        self.assertAlmostEqual(np.std(pca_features[:, 4]), 0.948, delta=0.2)

        sb_clf.fit(self.X_train, self.y_train_clf)
        mdi_feat_imp = feature_importance_mean_decrease_impurity(sb_clf, self.X_train.columns)
        pca_corr_res = feature_pca_analysis(self.X_train, mdi_feat_imp)

        # Check correlation metrics results
        self.assertAlmostEqual(pca_corr_res['Weighted_Kendall_Rank'][0], 0.26, delta=1e-1)

    def test_feature_importance(self):
        """
        Test features importance: MDI, MDA, SFI and plot function
        """
        sb_clf, cv_gen = self._prepare_clf_data_set(oob_score=False)

        # MDI feature importance
        mdi_feat_imp = feature_importance_mean_decrease_impurity(sb_clf, self.X_train.columns)

        # MDA feature importance
        mda_feat_imp_log_loss = feature_importance_mean_decrease_accuracy(sb_clf, self.X_train, self.y_train_clf,
                                                                          cv_gen,
                                                                          sample_weight=np.ones(
                                                                              (self.X_train.shape[0],)))
        mda_feat_imp_f1 = feature_importance_mean_decrease_accuracy(sb_clf, self.X_train, self.y_train_clf,
                                                                    cv_gen, scoring='f1')
        # SFI feature importance
        sfi_feat_imp_log_loss = feature_importance_sfi(sb_clf, self.X_train[self.X_train.columns[:5]], self.y_train_clf,
                                                       cv_gen=cv_gen, sample_weight=np.ones((self.X_train.shape[0],)))
        sfi_feat_imp_f1 = feature_importance_sfi(sb_clf, self.X_train[self.X_train.columns[:5]], self.y_train_clf,
                                                 cv_gen=cv_gen,
                                                 scoring='f1')  # Take only 5 features for faster test run

        # MDI assertions
        self.assertTrue(mdi_feat_imp['mean'].sum() == 1)
        # The most informative features
        self.assertAlmostEqual(mdi_feat_imp.loc['label_prob_0.1', 'mean'], 0.209, delta=0.01)
        self.assertAlmostEqual(mdi_feat_imp.loc['label_prob_0.2', 'mean'], 0.164, delta=0.01)
        # Noisy feature
        self.assertAlmostEqual(mdi_feat_imp.loc['label_prob_0.1_sma_5', 'mean'], 0.06253, delta=0.5)

        # MDA(log_loss) assertions
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['label_prob_0.1', 'mean'], 0.234, delta=0.3)
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['label_prob_0.2', 'mean'], 0.3222, delta=0.3)

        # MDA(f1) assertions
        self.assertAlmostEqual(mda_feat_imp_f1.loc['label_prob_0.1', 'mean'], 0.25, delta=0.3)
        self.assertAlmostEqual(mda_feat_imp_f1.loc['label_prob_0.2', 'mean'], 0.3, delta=0.3)
        self.assertLessEqual(mda_feat_imp_f1.loc['label_prob_0.1_sma_5', 'mean'], 0)

        # SFI(log_loss) assertions
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['label_prob_0.1', 'mean'], -2.14, delta=1)
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['label_prob_0.2', 'mean'], -2.15, delta=1)

        # SFI(accuracy) assertions
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['label_prob_0.1', 'mean'], 0.81, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['label_prob_0.2', 'mean'], 0.74, delta=1e-2)
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['label_prob_0.5_sma_2', 'mean'], 0.224, delta=1e-2)

    def test_plot_feature_importance(self):
        """
        Test plot_feature_importance function
        """

        sb_clf, cv_gen = self._prepare_clf_data_set(oob_score=True)
        oos_score = ml_cross_val_score(sb_clf, self.X_train, self.y_train_clf, cv_gen=cv_gen, sample_weight=None,
                                       scoring='accuracy').mean()

        sb_clf.fit(self.X_train, self.y_train_clf)

        mdi_feat_imp = feature_importance_mean_decrease_impurity(sb_clf, self.X_train.columns)
        plot_feature_importance(mdi_feat_imp, oob_score=sb_clf.oob_score_, oos_score=oos_score)
        plot_feature_importance(mdi_feat_imp, oob_score=sb_clf.oob_score_, oos_score=oos_score,
                                savefig=True, output_path='test.png')

        os.remove('test.png')

    def _prepare_clf_data_set(self, oob_score):
        """
        Helper function for preparing data sets for feature importance

        :param oob_score: (bool): bool flag for oob_score in classifier
        """
        clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                          class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_base, max_features=1.0, n_estimators=100,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=oob_score,
                                                           random_state=1)
        sb_clf.fit(self.X_train, self.y_train_clf)

        cv_gen = PurgedKFold(n_splits=4, samples_info_sets=self.samples_info_sets)
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
