"""
Implements the following classes from Chapter 12 of AFML:

- Combinatorial Purged Cross-Validation class.
- Stacked Combinatorial Purged Cross-Validation class.
"""
# pylint: disable=too-many-locals, arguments-differ, invalid-name, unused-argument

from itertools import combinations
from typing import List

import pandas as pd
import numpy as np
from scipy.special import comb
from sklearn.model_selection import KFold

from mlfinlab.cross_validation.cross_validation import ml_get_train_times


def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> int:
    """
    Number of combinatorial paths for CPCV(N,K).

    :param n_train_splits: (int) Number of train splits.
    :param n_test_splits: (int) Number of test splits.
    :return: (int) Number of backtest paths for CPCV(N,k).
    """

    pass


class CombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatorial Purged Cross Validation (CPCV).

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        pass

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        pass

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.
        """

        pass

    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X: (pd.DataFrame) Samples dataset that is to be split.
        :param y: (pd.Series) Sample labels series.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        pass


class StackedCombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Stacked Combinatorial Purged Cross Validation (CPCV). It implements CPCV for multiasset dataset.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets_dict: dict = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets_dict: (dict) Dictionary of samples info sets.
                                        ASSET_1: SAMPLE_INFO_SETS, ASSET_2:...

            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        pass

    def _fill_backtest_paths(self, asset, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param asset: (str) Asset for which backtest paths are filled.
        :param train_indices: (list) List of lists with first element corresponding to train start index, second - test end.
        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.
        """

        pass

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        pass

    def split(self,
              X_dict: dict,
              y_dict: dict = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X_dict: (dict) Dictionary of asset : X_{asset}.
        :param y_dict: (dict) Dictionary of asset : y_{asset}.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        pass
