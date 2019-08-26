"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import os
import unittest

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, mean_absolute_error, \
    mean_squared_error
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from mlfinlab.util.utils import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.sampling.bootstrapping import seq_bootstrap
from mlfinlab.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier, \
    SequentiallyBootstrappedBaggingRegressor


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

        self.X_train, self.y_train_clf, self.y_train_reg = X.iloc[:300][features], labels.iloc[:300].bin, \
                                                           labels.iloc[:300].ret
        self.X_test, self.y_test_clf, self.y_test_reg = X.iloc[300:][features], labels.iloc[300:].bin, \
                                                        labels.iloc[300:].ret

    def test_other_sb_features(self):
        """
        This function simply creates various estimators in order to get all lines of code covered, we don't check
        anything here
        """
        clf_1 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=True,
                                       class_weight='balanced_subsample', max_depth=12)
        clf_2 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                       class_weight='balanced_subsample', max_depth=12)
        clf_3 = KNeighborsClassifier()
        clf_4 = LinearSVC()

        sb_clf_1 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_1, max_features=0.2,
                                                             n_estimators=100,
                                                             triple_barrier_events=self.meta_labeled_events,
                                                             price_bars=self.data, oob_score=True,
                                                             random_state=1, bootstrap_features=True,
                                                             max_samples=30, verbose=2)

        sb_clf_2 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_2, max_features=7,
                                                             n_estimators=100,
                                                             triple_barrier_events=self.meta_labeled_events,
                                                             price_bars=self.data, oob_score=False,
                                                             random_state=1, bootstrap_features=True,
                                                             max_samples=0.3, warm_start=True)

        sb_clf_3 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_3,
                                                             triple_barrier_events=self.meta_labeled_events,
                                                             price_bars=self.data)

        sb_clf_4 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_4, max_features=0.2,
                                                             n_estimators=100,
                                                             triple_barrier_events=self.meta_labeled_events,
                                                             price_bars=self.data, oob_score=True,
                                                             random_state=1, bootstrap_features=True,
                                                             max_samples=30, verbose=2)

        sb_clf_1.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)), )

        sb_clf_2.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)), )
        sb_clf_2.n_estimators += 0
        sb_clf_2.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)), )
        sb_clf_2.n_estimators += 2
        sb_clf_2.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)), )

        sb_clf_3.fit(self.X_train, self.y_train_clf)
        sb_clf_4.fit(self.X_train, self.y_train_clf)

    def test_value_error_raise(self):
        """
        Test various values error raise
        """
        clf = KNeighborsClassifier()
        bagging_clf_1 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data)
        bagging_clf_2 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data, max_samples=2000000)
        bagging_clf_3 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data, max_features='20')
        bagging_clf_4 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data, max_features=2000000)
        bagging_clf_5 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data, oob_score=True, warm_start=True)
        bagging_clf_6 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data, warm_start=True)
        bagging_clf_7 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  triple_barrier_events=self.meta_labeled_events,
                                                                  price_bars=self.data, warm_start=True)
        with self.assertRaises(ValueError):
            # ValueError to use sample weight with classifier which doesn't support sample weights
            bagging_clf_1.fit(self.X_train, self.y_train_clf, sample_weight=np.ones((self.X_train.shape[0],)), )
        with self.assertRaises(ValueError):
            # ValueError for max_samples > X_train.shape[0]
            bagging_clf_2.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)), )
        with self.assertRaises(ValueError):
            # ValueError for non-int/float max_features param
            bagging_clf_3.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)), )
        with self.assertRaises(ValueError):
            # ValueError for max_features > X_train.shape[1]
            bagging_clf_4.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)), )
        with self.assertRaises(ValueError):
            # ValueError for warm_start and oob_score being True
            bagging_clf_5.fit(self.X_train, self.y_train_clf,
                              sample_weight=np.ones((self.X_train.shape[0],)), )
        with self.assertRaises(ValueError):
            # ValueError for decreasing the number of estimators when warm start is True
            bagging_clf_6.fit(self.X_train, self.y_train_clf)
            bagging_clf_6.n_estimators -= 2
            bagging_clf_6.fit(self.X_train, self.y_train_clf)
        with self.assertRaises(ValueError):
            # ValueError for setting n_estimators to negative value
            bagging_clf_7.fit(self.X_train, self.y_train_clf)
            bagging_clf_7.n_estimators -= 1000
            bagging_clf_7.fit(self.X_train, self.y_train_clf)

    def test_sb_classifier(self):
        """
        Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
        test oos predictions values
        """

        # Init classifiers
        clf = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                     class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf, max_features=1.0, n_estimators=100,
                                                           triple_barrier_events=self.meta_labeled_events,
                                                           price_bars=self.data, oob_score=True, random_state=1)
        sklearn_clf = BaggingClassifier(base_estimator=clf, max_features=1.0, n_estimators=50, oob_score=True,
                                        random_state=1)

        # X_train index should be in index mapping
        self.assertTrue(self.X_train.index.isin(sb_clf.timestamp_int_index_mapping.index).all())

        sb_clf.fit(self.X_train, self.y_train_clf)
        sklearn_clf.fit(self.X_train, self.y_train_clf)

        self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())  # X_train index == clf X_train index

        oos_sb_predictions = sb_clf.predict(self.X_test)
        oos_sklearn_predictions = sklearn_clf.predict(self.X_test)

        sb_precision = precision_score(self.y_test_clf, oos_sb_predictions)
        sb_recall = recall_score(self.y_test_clf, oos_sb_predictions)
        sb_roc_auc = roc_auc_score(self.y_test_clf, oos_sb_predictions)

        sklearn_precision = precision_score(self.y_test_clf, oos_sklearn_predictions)
        sklearn_recall = recall_score(self.y_test_clf, oos_sklearn_predictions)
        sklearn_roc_auc = roc_auc_score(self.y_test_clf, oos_sklearn_predictions)

        # Test OOB scores (sequentially bootstrapped, algorithm specific, standard (random sampling)

        # Algorithm specific
        self.assertGreater(sb_clf.oob_score_, sklearn_clf.oob_score_)  # oob_score for SB should be greater
        self.assertAlmostEqual(sb_clf.oob_score_, 0.99, delta=0.01)

        # Sequentially Bootstrapped oob_score
        # Trim index mapping so that only train indices are present
        subsamples = sb_clf.timestamp_int_index_mapping.loc[sb_clf.X_time_index]
        subsampled_ind_mat = sb_clf.ind_mat[:, subsamples]
        sb_sample = seq_bootstrap(subsampled_ind_mat, sample_length=self.X_train.shape[0], compare=True)
        sb_clf_accuracy = accuracy_score(self.y_train_clf.iloc[sb_sample],
                                         sb_clf.predict(self.X_train.iloc[sb_sample]))
        sklearn_clf_accuracy = accuracy_score(self.y_train_clf.iloc[sb_sample],
                                              sklearn_clf.predict(self.X_train.iloc[sb_sample]))
        self.assertGreaterEqual(sb_clf_accuracy, sklearn_clf_accuracy)

        # Random sampling oob_score
        random_sample = np.random.choice(subsampled_ind_mat.shape[1], size=self.X_train.shape[0])
        sb_clf_accuracy = accuracy_score(self.y_train_clf.iloc[random_sample],
                                         sb_clf.predict(self.X_train.iloc[sb_sample]))
        sklearn_clf_accuracy = accuracy_score(self.y_train_clf.iloc[random_sample],
                                              sklearn_clf.predict(self.X_train.iloc[sb_sample]))

        self.assertTrue(sb_clf_accuracy >= sklearn_clf_accuracy)

        # Test that OOS metrics for SB are greater than sklearn's
        self.assertGreaterEqual(sb_precision, sklearn_precision)
        self.assertGreaterEqual(sb_recall, sklearn_recall)
        self.assertGreaterEqual(sb_roc_auc, sklearn_roc_auc)

        self.assertTrue((oos_sb_predictions == np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0])).all())  # check oos predictions

    def test_sb_regressor(self):
        """
        Test Sequentially Bootstrapped Bagging Regressor
        """
        # Init regressors
        reg = RandomForestRegressor(n_estimators=1, bootstrap=False)
        sb_reg = SequentiallyBootstrappedBaggingRegressor(base_estimator=reg, max_features=1.0, n_estimators=100,
                                                          triple_barrier_events=self.meta_labeled_events,
                                                          price_bars=self.data, oob_score=True, random_state=1)

        sb_reg_70 = SequentiallyBootstrappedBaggingRegressor(base_estimator=reg, max_features=1.0, n_estimators=70,
                                                             triple_barrier_events=self.meta_labeled_events,
                                                             price_bars=self.data, oob_score=True, random_state=1)
        sb_reg_1_estimator = SequentiallyBootstrappedBaggingRegressor(base_estimator=reg, max_features=1.0,
                                                                      n_estimators=1,
                                                                      triple_barrier_events=self.meta_labeled_events,
                                                                      price_bars=self.data, oob_score=True,
                                                                      random_state=1)
        sklearn_reg = BaggingRegressor(base_estimator=reg, max_features=1.0, n_estimators=50, oob_score=True,
                                       random_state=1)

        # X_train index should be in index mapping

        sb_reg.fit(self.X_train, self.y_train_reg)
        sb_reg_1_estimator.fit(self.X_train, self.y_train_reg)
        sb_reg_70.fit(self.X_train.iloc[20:40], self.y_train_reg.iloc[20:40])  # Cover not raised oob warning
        sklearn_reg.fit(self.X_train, self.y_train_reg)

        self.assertTrue(self.X_train.index.isin(sb_reg.timestamp_int_index_mapping.index).all())
        self.assertTrue((sb_reg.X_time_index == self.X_train.index).all())  # X_train index == reg X_train index

        oos_sb_predictions = sb_reg.predict(self.X_test)
        oos_sklearn_predictions = sklearn_reg.predict(self.X_test)

        self.assertGreaterEqual(sb_reg.oob_score_, sklearn_reg.oob_score_)

        mse_sb_reg = mean_squared_error(self.y_test_reg, oos_sb_predictions)
        mae_sb_reg = mean_absolute_error(self.y_test_reg, oos_sb_predictions)

        mse_sklearn_reg = mean_squared_error(self.y_test_reg, oos_sklearn_predictions)
        mae_sklearn_reg = mean_absolute_error(self.y_test_reg, oos_sklearn_predictions)

        self.assertLessEqual(mae_sb_reg, mae_sklearn_reg)
        self.assertGreaterEqual(mse_sb_reg, mse_sklearn_reg)
