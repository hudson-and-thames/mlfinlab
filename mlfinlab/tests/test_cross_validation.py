"""
Tests the cross validation technique described in Ch.7 of the book
"""
import unittest
import os
import pandas as pd
import numpy as np

from mlfinlab.cross_validation.cross_validation import \
    get_train_times, \
    PurgedKFold


class TestCrossValidation(unittest.TestCase):  # pragma: no cover
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

        self.infosets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'),
        )
        self.log(self.infosets)

    def test_get_train_times_1(self):
        """
        Tests the get_train_times method for the case where the train STARTS within test
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:01:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = get_train_times(self.infosets, test_times)
        self.log(f"train_times_ret=\n{train_times_ret}")

        train_times_ok = pd.Series(
            index=pd.date_range(start='2019-01-01 00:03:00', end='2019-01-01 00:09:00', freq='T'),
            data=pd.date_range(start='2019-01-01 00:05:00', end='2019-01-01 00:11:00', freq='T'),
        )
        self.log(f"train_times=\n{train_times_ok}")

        self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")

    def test_get_train_times_2(self):
        """
        Tests the get_train_times method for the case where the train ENDS within test
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:08:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:11:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = get_train_times(self.infosets, test_times)
        self.log(f"train_times_ret=\n{train_times_ret}")

        train_times_ok = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', end='2019-01-01 00:05:00', freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', end='2019-01-01 00:07:00', freq='T'),
        )
        self.log(f"train_times=\n{train_times_ok}")

        self.assertTrue(train_times_ret.equals(train_times_ok), "train dataset doesn't match")

    def test_get_train_times_3(self):
        """
        Tests the get_train_times method for the case where the train ENVELOPES test
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:06:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:08:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = get_train_times(self.infosets, test_times)
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

    def test_purgedkfold_01(self):
        """
        Test throw exception when info_sets is not a pd.Series
        :return:
        """
        infosets = pd.DataFrame(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=20, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=20, freq='T'),
        )
        self.log(f"infosets=\n{infosets}")

        dataset = pd.DataFrame(
            index=infosets.index,
            data={'feat': np.arange(0, 20)},
        )
        self.log(f"dataset=\n{dataset}")

        try:
            PurgedKFold(n_splits=3, info_sets=infosets, pct_embargo=0.)
        except ValueError:
            pass
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")
        else:
            self.fail("ValueError not raised")
        self.assertTrue(True)

    def test_purgedkfold_02(self):
        """
        Test exception is raised when passing in a dataset with a different index
        than the infosets used in the constructor
        :return:
        """
        infosets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'),
        )
        self.log(f"infosets=\n{infosets}")

        dataset = pd.DataFrame(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=11, freq='T'),
            data={'feat': np.arange(0, 11)},  # one entry more than infosets
        )
        self.log(f"dataset=\n{dataset}")

        pkf = PurgedKFold(n_splits=3, info_sets=infosets, pct_embargo=0.)
        try:
            for _, _ in pkf.split(dataset):
                pass
        except ValueError:
            pass
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")
        else:
            self.fail("ValueError not raised")
        self.assertTrue(True)

    def test_purgedkfold_1(self):
        """
        Test PurgedKFold class using the get_train_times method. Get the test range from PurgedKFold
        and then make sure the train range is exactly the same using the two methods.
        This is the test with no embargo.
        :return:
        """

        infosets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=20, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=20, freq='T'),
        )
        self.log(f"infosets=\n{infosets}")

        dataset = pd.DataFrame(
            index=infosets.index,
            data={'feat': np.arange(0, 20)},
        )
        self.log(f"dataset=\n{dataset}")

        pkf = PurgedKFold(n_splits=3, info_sets=infosets, pct_embargo=0.)
        for train_indices, test_indices in pkf.split(dataset):
            self.log(f"test_times_ret=\n{infosets[test_indices]}")

            train_times_ret = infosets.iloc[train_indices]
            self.log(f"train_times_ret=\n{train_times_ret}")

            test_times_gtt = pd.Series(
                index=[infosets[test_indices[0]]],
                data=[infosets[test_indices[-1]]],
            )

            self.log(f"test_times_gtt=\n{test_times_gtt}")
            train_times_gtt = get_train_times(infosets, test_times_gtt)
            self.log(f"train_times_gtt=\n{train_times_gtt}")
            self.log("-" * 100)

            self.assertTrue(train_times_ret.equals(train_times_gtt), "dataset don't match")
