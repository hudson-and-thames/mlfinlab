"""
Tests the cross validation technique described in Ch.7 of the book
"""
import unittest
import os
import pandas as pd

from mlfinlab.cross_validation.cross_validation import get_train_times


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

        self.observations = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=10, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=10, freq='T'),
        )
        self.log(self.observations)

    def test_get_train_times_1(self):
        """
        Tests the get_train_times method for the case where the train STARTS within test
        """
        test_times = pd.Series(
            index=pd.date_range(start='2019-01-01 00:01:00', periods=1, freq='T'),
            data=pd.date_range(start='2019-01-01 00:02:00', periods=1, freq='T'),
        )
        self.log(f"test_times=\n{test_times}")
        train_times_ret = get_train_times(self.observations, test_times)
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
        train_times_ret = get_train_times(self.observations, test_times)
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
        train_times_ret = get_train_times(self.observations, test_times)
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
