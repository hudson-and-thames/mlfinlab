"""
Tests the cross validation technique described in Ch.7 of the book.
"""
import unittest
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score

from mlfinlab.cross_validation.cross_validation import (
    ml_get_train_times,
    ml_cross_val_score,
    PurgedKFold
)


class TestCrossValidation(unittest.TestCase):
    """
    Test the functionality of the time series cross validation technique
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.verbose = False

    def log(self, msg):
        """
        Simple method to suppress debugging strings
        """
        if self.verbose:  # pragma: no cover
            print(msg)

    def setUp(self):
        """
        This is how the observations dataset looks like
        2019-01-01 00:00:00   2019-01-01 00:02:00
        2019-01-01 00:01:00   2019-01-01 00:03:00
        2019-01-01 00:02:00   2019-01-01 00:04:00
        2019-01-01 00:03:00   2019-01-01 00:05:00
        2019-01-01 00:04:00   2019-01-01 00:06:00
        2019-01-01 00:05:00   2019-01-01 00:07:00
        2019-01-01 00:06:00   2019-01-01 00:08:00
        2019-01-01 00:07:00   2019-01-01 00:09:00
        2019-01-01 00:08:00   2019-01-01 00:10:00
        2019-01-01 00:09:00   2019-01-01 00:11:00
        """

        pwd_path = os.path.dirname(__file__)
        self.log(f"pwd_path= {pwd_path}")

        self.info_sets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'),
        )

    def test_get_train_times_1(self):
        """
        Tests the get_train_times method for the case where the train STARTS within test.
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:01:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = ml_get_train_times(self.info_sets, test_times)
        self.log(f"train_times_ret=\n{train_times_ret}")

        train_times_ok = pd.Series(
            index=pd.date_range(start='2019-01-01 00:03:00', end='2019-01-01 00:09:00', freq='T'),
            data=pd.date_range(start='2019-01-01 00:05:00', end='2019-01-01 00:11:00', freq='T'),
        )
        self.log(f"train_times=\n{train_times_ok}")

        self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")

    def test_get_train_times_2(self):
        """
        Tests the get_train_times method for the case where the train ENDS within test.
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:08:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:11:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = ml_get_train_times(self.info_sets, test_times)
        self.log(f"train_times_ret=\n{train_times_ret}")

        train_times_ok = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', end='2019-01-01 00:05:00', freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', end='2019-01-01 00:07:00', freq='T'),
        )
        self.log(f"train_times=\n{train_times_ok}")

        self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")

    def test_get_train_times_3(self):
        """
        Tests the get_train_times method for the case where the train ENVELOPES test.
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:06:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:08:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = ml_get_train_times(self.info_sets, test_times)
        self.log(f"train_times_ret=\n{train_times_ret}")

        train_times_ok1 = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', end='2019-01-01 00:03:00', freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', end='2019-01-01 00:05:00', freq='T'),
        )
        train_times_ok2 = pd.Series(
            index=pd.date_range(start='2019-01-01 00:09:00', end='2019-01-01 00:09:00', freq='T'),
            data=pd.date_range(start='2019-01-01 00:11:00', end='2019-01-01 00:11:00', freq='T'),
        )
        train_times_ok = pd.concat([train_times_ok1, train_times_ok2])
        self.log(f"train_times=\n{train_times_ok}")

        self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")

    def test_purgedkfold_01_exception(self):
        """
        Test throw exception when samples_info_sets is not a pd.Series.
        """
        samples_info_sets = pd.DataFrame(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=20, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=20, freq='T'),
        )
        self.log(f"info_sets=\n{samples_info_sets}")

        dataset = pd.DataFrame(
            index=samples_info_sets.index,
            data={'feat': np.arange(0, 20)},
        )
        self.log(f"dataset=\n{dataset}")

        try:
            PurgedKFold(n_splits=3, samples_info_sets=samples_info_sets, pct_embargo=0.)
        except ValueError:
            pass
        else:
            self.fail("ValueError not raised")

    def test_purgedkfold_02_exception(self):
        """
        Test exception is raised when passing in a dataset with a different length than the samples_info_sets used in the
        constructor.
        """
        info_sets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'),
        )
        self.log(f"info_sets=\n{info_sets}")

        dataset = pd.DataFrame(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=11, freq='T'),
            data={'feat': np.arange(0, 11)},  # One entry more than info_sets
        )
        self.log(f"dataset=\n{dataset}")

        pkf = PurgedKFold(n_splits=3, samples_info_sets=info_sets, pct_embargo=0.)
        try:
            for _, _ in pkf.split(dataset):
                pass
        except ValueError:
            pass
        else:
            self.fail("ValueError not raised")

    def test_purgedkfold_03_simple(self):
        """
        Test PurgedKFold class using the ml_get_train_times method. Get the test range from PurgedKFold and then make
        sure the train range is exactly the same using the two methods.

        This is the test with no embargo.
        """

        info_sets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=20, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=20, freq='T'),
        )
        self.log(f"info_sets=\n{info_sets}")

        dataset = pd.DataFrame(
            index=info_sets.index,
            data={'feat': np.arange(0, 20)},
        )
        self.log(f"dataset=\n{dataset}")

        pkf = PurgedKFold(n_splits=3, samples_info_sets=info_sets, pct_embargo=0.)
        for train_indices, test_indices in pkf.split(dataset):
            self.log(f"test_times_ret=\n{info_sets[test_indices]}")

            train_times_ret = info_sets.iloc[train_indices]
            self.log(f"train_times_ret=\n{train_times_ret}")

            test_times_gtt = pd.Series(
                index=[info_sets[test_indices[0]]],
                data=[info_sets[test_indices[-1]]],
            )

            self.log(f"test_times_gtt=\n{test_times_gtt}")
            train_times_gtt = ml_get_train_times(info_sets, test_times_gtt)
            self.log(f"train_times_gtt=\n{train_times_gtt}")
            self.log("-" * 100)

            self.assertTrue(train_times_ret.equals(train_times_gtt), "dataset don't match")

    def test_purgedkfold_04_embargo(self):
        """
        Test PurgedKFold class using the 'embargo' parameter set to pct_points_test which means pct_points_test percent
        which also means pct_points_test entries from a total of 100 in total in the dataset.
        """

        info_sets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=100, freq='T'),
        )

        dataset = pd.DataFrame(
            index=info_sets.index,
            data={'feat': np.arange(0, 100)},
        )
        pct_points_test: int = 2
        self.log(f"pct_points_test= {pct_points_test}")

        pkf = PurgedKFold(n_splits=3, samples_info_sets=info_sets, pct_embargo=0.01*pct_points_test)

        # Also test that X can be an np.ndarray by passing in the .values of the pd.DataFrame
        for train_indices, test_indices in pkf.split(dataset):

            train_times_ret = info_sets.iloc[train_indices]
            self.log(f"train_times_ret=\n{train_times_ret}")

            test_times_ret = info_sets.iloc[test_indices]
            self.log(f"test_times_ret=\n{test_times_ret}")

            test_times_gtt = pd.Series(
                index=[info_sets[test_indices[0]]],
                data=[info_sets[test_indices[-1]]],
            )
            self.log(f"test_times_gtt=\n{test_times_gtt}")

            train_times_gtt = ml_get_train_times(info_sets, test_times_gtt)

            # if test set is at the beginning, drop pct_points_test records from beginning of train dataset
            if test_times_ret.index[0] == dataset.index[0]:
                train_times_gtt = train_times_gtt.iloc[pct_points_test:]

            # if test set is in the middle drop pct_points_test records from the end of test set index
            elif test_times_ret.index[-1] != dataset.index[-1]:
                last_test_ix = test_times_ret.index.get_loc(test_times_ret.index[-1])
                to_drop: pd.DatetimeIndex = train_times_gtt.iloc[last_test_ix+2:last_test_ix+2+pct_points_test].index
                train_times_gtt.drop(to_drop.to_list(), inplace=True)

            self.log(f"train_times_gtt=\n{train_times_gtt}")
            self.log("-" * 100)
            self.assertTrue(train_times_ret.equals(train_times_gtt), "dataset don't match")

    def _test_ml_cross_val_score__data(self):
        """
        Get data structures for next few tests.
        """
        sample_size = 1000

        info_sets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=sample_size, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=sample_size, freq='T'),
        )

        records = pd.DataFrame(
            index=info_sets.index,
            data={
                'even': np.arange(0, sample_size),
                'odd': np.arange(1, sample_size+1)
            },
        )
        labels = pd.Series(
            index=info_sets.index,
            data=np.arange(0, sample_size)
        )
        labels[records['even'] % 2 == 0] = 1
        labels[records['even'] % 2 != 0] = 0
        self.log(f"y=\n{labels[:10]}")

        random_state = np.random.RandomState(seed=12345)
        sample_weights = pd.Series(
            index=info_sets.index,
            data=random_state.random_sample(sample_size)
        )

        decision_tree = DecisionTreeClassifier(random_state=0)
        return info_sets, records, labels, sample_weights, decision_tree

    def test_ml_cross_val_score_01_accuracy(self):
        """
        Test the ml_cross_val_score function with an artificial dataset.
        """
        info_sets, records, labels, sample_weights, decision_tree = self._test_ml_cross_val_score__data()
        cv_gen = PurgedKFold(samples_info_sets=info_sets, n_splits=3, pct_embargo=0.01)
        scores = ml_cross_val_score(
            classifier=decision_tree,
            X=records,
            y=labels,
            sample_weight_train=sample_weights.values,
            sample_weight_score=sample_weights.values,
            scoring=accuracy_score,
            cv_gen=cv_gen,
        )
        self.log(f"score1= {scores}")

        should_be = np.array([0.5186980141893885, 0.4876916232189882, 0.4966185791847402])
        self.assertTrue(
            np.array_equal(scores, should_be),
            "score lists don't match"
        )

    def test_ml_cross_val_score_02_neg_log_loss(self):
        """
        Test the ml_cross_val_score function with an artificial dataset.
        """
        info_sets, records, labels, sample_weights, decision_tree = self._test_ml_cross_val_score__data()
        cv_gen = PurgedKFold(samples_info_sets=info_sets, n_splits=3, pct_embargo=0.01)
        scores = ml_cross_val_score(
            classifier=decision_tree,
            X=records,
            y=labels,
            sample_weight_train=sample_weights.values,
            sample_weight_score=None,
            scoring=log_loss,
            cv_gen=cv_gen,
        )
        self.log(f"scores= {scores}")

        should_be = np.array([-17.26939, -17.32125, -17.32125])
        self.assertTrue(
            np.allclose(scores, should_be),
            "score lists don't match"
        )

    def test_ml_cross_val_score_03_other_cv_gen(self):
        """
        Test the ml_cross_val_score function with an artificial dataset.
        """
        _, records, labels, sample_weights, decision_tree = self._test_ml_cross_val_score__data()
        scores = ml_cross_val_score(
            classifier=decision_tree,
            X=records,
            y=labels,
            sample_weight_train=sample_weights.values,
            sample_weight_score=sample_weights.values,
            scoring=log_loss,
            cv_gen=TimeSeriesSplit(max_train_size=None, n_splits=3),
        )
        self.log(f"scores= {scores}")

        should_be = np.array([-17.520701311460694, -18.25536255165772, -16.964650471071668])
        self.assertTrue(
            np.array_equal(scores, should_be),
            # self.assertListEqual(scores.tolist(), should_be.tolist()),
            "score lists don't match"
        )

    def test_ml_cross_val_score_04_sw(self):
        """
        Test the ml_cross_val_score function with an artificial dataset.
        """
        info_sets, records, labels, _, decision_tree = self._test_ml_cross_val_score__data()
        cv_gen = PurgedKFold(samples_info_sets=info_sets, n_splits=3, pct_embargo=0.01)
        scores = ml_cross_val_score(
            classifier=decision_tree,
            X=records,
            y=labels,
            sample_weight_train=None,
            sample_weight_score=None,
            scoring=accuracy_score,
            cv_gen=cv_gen,
        )
        self.log(f"score1= {scores}")

        should_be = np.array([0.5, 0.4984984984984985, 0.4984984984984985])
        self.assertTrue(
            np.array_equal(scores, should_be),
            "score lists don't match"
        )
