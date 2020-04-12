"""
Tests the cross validation technique described in Ch.7 of the book.
"""
import unittest
import pandas as pd
from mlfinlab.cross_validation import CombinatorialPurgedKFold


class TestCombinatorialPurgedCV(unittest.TestCase):
    """
    Test the functionality of the time series cross validation technique
    """

    def setUp(self):
        """
        Test Combinatorial Purged CV class
        """

        self.info_sets = pd.Series(
            index=pd.date_range(start='2019-01-01 00:00:00', periods=100, freq='D'),
            data=pd.date_range(start='2019-01-02 00:02:00', periods=100, freq='T'),
        )

    def test_test_times(self):
        """
        Tests the Combinatorial CV with various number of train, test splits
        """
        cv_gen = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=self.info_sets)
        counter = 0  # number of splits counter
        train_splits = []  # Array of train indices splits
        test_splits = []  # Array of test indices splits

        for train, test in cv_gen.split(self.info_sets):
            counter += 1
            train_splits.append(train)
            test_splits.append(test)

        self.assertEqual(counter, 15)  # Example from the book says that for CombinatorialCV(6, 2) number of splits=15

        # Check indices from example from the book (page 164)
        for test_idx, [start_idx, end_idx] in zip(test_splits[0:5], [[0, 33], [0, 50], [0, 67], [0, 83], [0, 99]]):
            self.assertEqual(test_idx[0], start_idx)
            self.assertEqual(test_idx[-1], end_idx)

        for test_idx, [start_idx, end_idx] in zip(test_splits[-5:], [[34, 83], [34, 99], [51, 83], [51, 99], [68, 99]]):
            self.assertEqual(test_idx[0], start_idx)
            self.assertEqual(test_idx[-1], end_idx)

    def test_purge_and_embargo(self):
        """
        Test that purging and embargo works correctly
        """

        cv_gen = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=self.info_sets,
                                          pct_embargo=0.05)
        train_splits = []
        test_splits = []

        for train, test in cv_gen.split(self.info_sets):
            train_splits.append(train)
            test_splits.append(test)

        cv_gen_no_embargo = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=self.info_sets,
                                                     pct_embargo=0.0)
        train_splits_no = []
        test_splits_no = []

        for train, test in cv_gen_no_embargo.split(self.info_sets):
            train_splits_no.append(train)
            test_splits_no.append(test)

        self.assertEqual(len(train_splits_no[0]) - len(train_splits[0]), 5)  # Embargo of 1 train set
        self.assertEqual(len(train_splits_no[1]) - len(train_splits[1]), 10)  # Embargo of 2 train set
        self.assertEqual(len(train_splits_no[-1]) - len(train_splits[-1]), 0)  # Latest set is not embargo-ed

    def test_errors_raise(self):
        """
        Test if ValueErrors are raised
        """

        with self.assertRaises(ValueError):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=3, samples_info_sets=[1, 2, 3, 4])

        cv_gen = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=self.info_sets,
                                          pct_embargo=0.05)

        with self.assertRaises(ValueError):
            for _, _ in cv_gen.split(self.info_sets.iloc[2:]):
                pass
