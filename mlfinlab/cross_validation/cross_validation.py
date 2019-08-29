"""
Implements the book chapter 7 on Cross Validation for financial data
"""

import pandas as pd
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator


def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    # pylint: disable=invalid-name
    """
    Snippet 7.1, page 106,  Purging observations in the training set

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.

    :param samples_info_sets: The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param test_times: Times for the test dataset.
    """
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.iteritems():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # Train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # Train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals
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
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

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

        indices: np.ndarray = np.arange(X.shape[0])
        embargo: int = int(X.shape[0] * self.pct_embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(index=[self.samples_info_sets[start_ix]], data=[self.samples_info_sets[end_ix-1]])
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            yield np.array(train_indices), test_indices


# noinspection PyPep8Naming
def ml_cross_val_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray,
        scoring: str = 'neg_log_loss',
        samples_info_sets: pd.Series = None,
        n_splits: int = None,
        cv_gen: BaseCrossValidator = None,
        pct_embargo: int = None):
    # pylint: disable=invalid-name
    """
    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator

    :param classifier: A sk-learn Classifier object instance
    :param X: The dataset of records to evaluate
    :param y: The labels corresponding to the X dataset
    :param sample_weight: A numpy array of weights for each record in the dataset
    :param scoring: A metric name to use for scoring; currently supports `neg_log_loss` and `accuracy`
    :param samples_info_sets: The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param n_splits: Number of splits
    :param cv_gen: Cross Validation generator object instance; if None then PurgedKFold will be used
    :param pct_embargo: Embargo percentage [0, 1]
    :return: The computed score
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise ValueError('wrong scoring method.')
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=n_splits, samples_info_sets=samples_info_sets, pct_embargo=pct_embargo)
    ret_scores = []
    for train, test in cv_gen.split(X=X):
        fit = classifier.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score = -1*log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=classifier.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        ret_scores.append(score)
    return np.array(ret_scores)
