"""
Implements the book chapter 7 on Cross Validation for financial data
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold


def get_train_times(info_sets: pd.Series, test_times: pd.Series) -> pd.Series:  # pragma: no cover
    """
    Given test_times, find the times of the training observations.
    —overlap.index: Time when the information extraction started.
    —overlap.value: Time when the information extraction ended.
    —test_times: Times for the test dataset.
    """
    train = info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.iteritems():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """

    def __init__(self, n_splits=3, info_sets=None, pct_embargo=0.):
        """
        :param n_splits: The number of splits. Default to 3
        :param info_sets:
            —overlap.index: Time when the information extraction started.
            —overlap.value: Time when the information extraction ended.
        :param pct_embargo: Percent that determines the embargo size.
        """
        if not isinstance(info_sets, pd.Series):
            raise ValueError('The info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.info_sets = info_sets
        self.pct_embargo = pct_embargo

    # noinspection PyPep8Naming
    def split(self, X, y=None, groups=None):
        if len(X.index) != len(self.info_sets.index) or \
                (X.index == self.info_sets.index).sum() != len(self.info_sets):
            raise ValueError('X and the `info_sets` series param must have the same index')

        indices: np.ndarray = np.arange(X.shape[0])
        embargo: int = int(X.shape[0] * self.pct_embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(index=[self.info_sets[start_ix]], data=[self.info_sets[end_ix-1]])
            train_times = get_train_times(self.info_sets, test_times)

            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.info_sets.index.get_loc(train_ix))
            yield train_indices, test_indices
