"""
Implements the Combinatorial Purged Cross-Validation class from Chapter 12
"""

from itertools import combinations
from typing import List

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from .cross_validation import ml_get_train_times


class CombinatorialPurgedKFold(KFold):
    """
    Implements Combinatial Purged Cross Validation (CPCV) from Chapter 12
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between

    :param n_splits: The number of splits. Default to 3
    :param samples_info_sets: The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param pct_embargo: Percent that determines the embargo size.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(CombinatorialPurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.n_test_splits = n_test_splits
        self.backtest_paths = {k: list(range(self.n_splits)) for k in
                               range(self.n_splits - 1)}  # Dictionary of backtest paths, number of paths = n_splits - 1

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits

        :param splits_indices: (dict) of test fold integer index: [start test index, end test index]
        :return: (list) of combinatorial test splits ([start index, end index])
        """

        # Possible test splits for each fold
        combinatorial_splits = list(combinations(list(splits_indices.keys()), self.n_test_splits))
        combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = []  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)
        return combinatorial_test_ranges

    # noinspection PyPep8Naming
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None):
        """
        The main method to call for the PurgedKFold class

        :param X: The pd.DataFrame samples dataset that is to be split
        :param y: The pd.Series sample labels series
        :param groups: array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: This method yields uples of (train, test) where train and test are lists of sample indices
        """
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        splits_indices = {}
        for index, [start_ix, end_ix] in enumerate(test_ranges):
            splits_indices[index] = [start_ix, end_ix]

        combinatorial_test_ranges = self._generate_combinatorial_test_ranges(splits_indices)

        embargo: int = int(X.shape[0] * self.pct_embargo)
        for test_splits in combinatorial_test_ranges:

            # Embargo
            test_times = pd.Series(index=[self.samples_info_sets[ix[0]] for ix in test_splits], data=[
                self.samples_info_sets[ix[1] - 1] if ix[1] - 1 + embargo >= X.shape[0] else self.samples_info_sets[
                    ix[1] - 1 + embargo]
                for ix in test_splits])

            test_indices = []
            for [start_ix, end_ix] in test_splits:
                test_indices.extend(list(range(start_ix, end_ix)))

            # Purge
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            # Get indices
            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            yield np.array(train_indices), np.array(test_indices)
