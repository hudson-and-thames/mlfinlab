"""
Test various functions regarding chapter 4: Sampling (Bootstrapping, Concurrency).
"""

import os
import unittest

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, mean_absolute_error, \
    mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import indices_to_mask

from mlfinlab.util.utils import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix, get_ind_mat_average_uniqueness, \
    get_ind_mat_label_uniqueness
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

        # Get mix of samples where some of them are extremely non-overlapping, the other one are highly overlapping
        good_uniqueness_thresh = 0.5
        bad_uniqueness_thresh = 0.1
        i = 0
        unique_samples = []
        for label in get_ind_mat_label_uniqueness(ind_mat):
            if np.mean(label[label > 0]) > good_uniqueness_thresh or np.mean(label[label > 0]) < bad_uniqueness_thresh:
                unique_samples.append(i)
            i += 1

        X = self.data.loc[labels.index,].iloc[unique_samples].dropna()  # get synthetic data set with drawn samples
        labels = labels.loc[X.index, :]
        X.loc[labels.index, 'y'] = labels.bin

        # Generate features (some of them are informative, others are just noise)
        for index, value in X.y.iteritems():
            X.loc[index, 'label_prob_0.6'] = self._generate_label_with_prob(value, 0.6)
            X.loc[index, 'label_prob_0.5'] = self._generate_label_with_prob(value, 0.5)
            X.loc[index, 'label_prob_0.3'] = self._generate_label_with_prob(value, 0.3)
            X.loc[index, 'label_prob_0.2'] = self._generate_label_with_prob(value, 0.2)
            X.loc[index, 'label_prob_0.1'] = self._generate_label_with_prob(value, 0.1)

        features = ['label_prob_0.6', 'label_prob_0.2']  # Two super-informative features
        for prob in [0.5, 0.3, 0.2, 0.1]:
            for window in [2, 5, 10]:
                X['label_prob_{}_sma_{}'.format(prob, window)] = X['label_prob_{}'.format(prob)].rolling(
                    window=window).mean()
                features.append('label_prob_{}_sma_{}'.format(prob, window))
        X.dropna(inplace=True)
        y = X.pop('y')

        self.X_train, self.X_test, self.y_train_clf, self.y_test_clf = train_test_split(X[features], y, test_size=0.4,
                                                                                        random_state=1, shuffle=False)
        self.y_train_reg = (1 + self.y_train_clf) * np.random.random_sample()
        self.y_test_reg = (1 + self.y_test_clf) * np.random.random_sample()

        self.samples_info_sets = meta_labeled_events.loc[self.X_train.index, 't1']
        self.price_bars_trim = self.data[
            (self.data.index >= self.X_train.index.min()) & (self.data.index <= self.X_train.index.max())].close

    def _generate_label_with_prob(self, x, prob, random_state=np.random.RandomState(1)):
        """
        Generates true label value with some probability(prob)
        """
        choice = random_state.choice([0, 1], p=[1 - prob, prob])
        if choice == 1:
            return x
        else:
            return int(not x)

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
                                                             samples_info_sets=self.meta_labeled_events.t1,
                                                             price_bars=self.data, oob_score=True,
                                                             random_state=1, bootstrap_features=True,
                                                             max_samples=30, verbose=2)

        sb_clf_2 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_2, max_features=7,
                                                             n_estimators=100,
                                                             samples_info_sets=self.meta_labeled_events.t1,
                                                             price_bars=self.data, oob_score=False,
                                                             random_state=1, bootstrap_features=True,
                                                             max_samples=0.3, warm_start=True)

        sb_clf_3 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_3,
                                                             samples_info_sets=self.meta_labeled_events.t1,
                                                             price_bars=self.data)

        sb_clf_4 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_4, max_features=0.2,
                                                             n_estimators=100,
                                                             samples_info_sets=self.meta_labeled_events.t1,
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
                                                                  samples_info_sets=self.meta_labeled_events.t1,
                                                                  price_bars=self.data)
        bagging_clf_2 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.meta_labeled_events.t1,
                                                                  price_bars=self.data, max_samples=2000000)
        bagging_clf_3 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.meta_labeled_events.t1,
                                                                  price_bars=self.data, max_features='20')
        bagging_clf_4 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.meta_labeled_events.t1,
                                                                  price_bars=self.data, max_features=2000000)
        bagging_clf_5 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.meta_labeled_events.t1,
                                                                  price_bars=self.data, oob_score=True, warm_start=True)
        bagging_clf_6 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.meta_labeled_events.t1,
                                                                  price_bars=self.data, warm_start=True)
        bagging_clf_7 = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf,
                                                                  samples_info_sets=self.meta_labeled_events.t1,
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
        clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                          class_weight='balanced_subsample')

        sb_clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=clf_base, max_features=1.0, n_estimators=100,
                                                           samples_info_sets=self.samples_info_sets,
                                                           price_bars=self.price_bars_trim, oob_score=True,
                                                           random_state=1)

        # X_train index should be in index mapping
        self.assertTrue(self.X_train.index.isin(sb_clf.timestamp_int_index_mapping.index).all())

        sb_clf.fit(self.X_train, self.y_train_clf)

        self.assertTrue((sb_clf.X_time_index == self.X_train.index).all())  # X_train index == clf X_train index

        oos_sb_predictions = sb_clf.predict(self.X_test)

        sb_precision = precision_score(self.y_test_clf, oos_sb_predictions)
        sb_recall = recall_score(self.y_test_clf, oos_sb_predictions)
        sb_roc_auc = roc_auc_score(self.y_test_clf, oos_sb_predictions)
        sb_accuracy = accuracy_score(self.y_test_clf, oos_sb_predictions)

        self.assertAlmostEqual(sb_accuracy, 0.66, delta=0.01)
        self.assertEqual(sb_precision, 1.0)
        self.assertAlmostEqual(sb_recall, 0.18, delta=0.01)
        self.assertAlmostEqual(sb_roc_auc, 0.59, delta=0.01)

        # Test OOB score
        self.assertTrue((oos_sb_predictions == np.array(
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0])).all())  # check oos predictions

    def test_sb_regressor(self):
        """
        Test Sequentially Bootstrapped Bagging Regressor
        """
        # Init regressors
        reg = RandomForestRegressor(n_estimators=1, bootstrap=False)
        sb_reg = SequentiallyBootstrappedBaggingRegressor(base_estimator=reg, max_features=1.0, n_estimators=100,
                                                          samples_info_sets=self.samples_info_sets,
                                                          price_bars=self.price_bars_trim, oob_score=True,
                                                          random_state=1)

        # X_train index should be in index mapping

        sb_reg.fit(self.X_train, self.y_train_reg)

        self.assertTrue(self.X_train.index.isin(sb_reg.timestamp_int_index_mapping.index).all())
        self.assertTrue((sb_reg.X_time_index == self.X_train.index).all())  # X_train index == reg X_train index

        oos_sb_predictions = sb_reg.predict(self.X_test)
        mse_sb_reg = mean_squared_error(self.y_test_reg, oos_sb_predictions)
        mae_sb_reg = mean_absolute_error(self.y_test_reg, oos_sb_predictions)

        self.assertAlmostEqual(mse_sb_reg, 0.7, delta=0.01)
        self.assertAlmostEqual(mae_sb_reg, 0.767, delta=0.01)
