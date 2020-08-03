"""
Implements the book chapter 7 on Cross Validation for financial data.
"""

from typing import Callable
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator


def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    # pylint: disable=invalid-name
    """
    Advances in Financial Machine Learning, Snippet 7.1, page 106.

    Purging observations in the training set

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.

    :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param test_times: (pd.Series) Times for the test dataset.
    :return: (pd.Series) Training set
    """

    pass


class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals

    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between

    :param n_splits: (int) The number of splits. Default to 3
    :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param pct_embargo: (float) Percent that determines the embargo size.
    """

    def __init__(self,
                 n_splits: int = 3,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):


        pass

    # noinspection PyPep8Naming
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None):
        """
        The main method to call for the PurgedKFold class

        :param X: (pd.DataFrame) Samples dataset that is to be split
        :param y: (pd.Series) Sample labels series
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices]
        """

        pass


# noinspection PyPep8Naming
def ml_cross_val_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        cv_gen: BaseCrossValidator,
        sample_weight_train: np.ndarray = None,
        sample_weight_score: np.ndarray = None,
        scoring: Callable[[np.array, np.array], float] = log_loss):
    # pylint: disable=invalid-name
    # pylint: disable=comparison-with-callable
    """
    Advances in Financial Machine Learning, Snippet 7.4, page 110.

    Using the PurgedKFold Class.

    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator.

    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:

    .. code-block:: python

        cv_gen = PurgedKFold(n_splits=n_splits, samples_info_sets=samples_info_sets, pct_embargo=pct_embargo)
        scores_array = ml_cross_val_score(classifier, X, y, cv_gen, sample_weight_train=sample_train,
                                          sample_weight_score=sample_score, scoring=accuracy_score)

    :param classifier: (ClassifierMixin) A sk-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (Callable) A metric scoring, can be custom sklearn metric.
    :return: (np.array) The computed score.
    """

    pass
