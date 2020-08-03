"""
Logic regarding sequential bootstrapping from chapter 4.
"""

import pandas as pd
import numpy as np
from numba import jit, prange


def get_ind_matrix(samples_info_sets, price_bars):
    """
    Advances in Financial Machine Learning, Snippet 4.3, page 65.

    Build an Indicator Matrix

    Get indicator matrix. The book implementation uses bar_index as input, however there is no explanation
    how to form it. We decided that using triple_barrier_events and price bars by analogy with concurrency
    is the best option.

    :param samples_info_sets: (pd.Series): Triple barrier events(t1) from labeling.get_events
    :param price_bars: (pd.DataFrame): Price bars which were used to form triple barrier events
    :return: (np.array) Indicator binary matrix indicating what (price) bars influence the label for each observation
    """

    pass


def get_ind_mat_average_uniqueness(ind_mat):
    """
    Advances in Financial Machine Learning, Snippet 4.4. page 65.

    Compute Average Uniqueness

    Average uniqueness from indicator matrix

    :param ind_mat: (np.matrix) Indicator binary matrix
    :return: (float) Average uniqueness
    """

    pass


def get_ind_mat_label_uniqueness(ind_mat):
    """
    Advances in Financial Machine Learning, An adaption of Snippet 4.4. page 65.

    Returns the indicator matrix element uniqueness.

    :param ind_mat: (np.matrix) Indicator binary matrix
    :return: (np.matrix) Element uniqueness
    """

    pass


@jit(parallel=True, nopython=True)
def _bootstrap_loop_run(ind_mat, prev_concurrency):  # pragma: no cover
    """
    Part of Sequential Bootstrapping for-loop. Using previously accumulated concurrency array, loops through all samples
    and generates averages uniqueness array of label based on previously accumulated concurrency

    :param ind_mat (np.array): Indicator matrix from get_ind_matrix function
    :param prev_concurrency (np.array): Accumulated concurrency from previous iterations of sequential bootstrapping
    :return: (np.array): Label average uniqueness based on prev_concurrency
    """

    pass


def seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None, compare=False, verbose=False,
                  random_state=np.random.RandomState()):
    """
    Advances in Financial Machine Learning, Snippet 4.5, Snippet 4.6, page 65.

    Return Sample from Sequential Bootstrap

    Generate a sample via sequential bootstrap.
    Note: Moved from pd.DataFrame to np.matrix for performance increase

    :param ind_mat: (pd.DataFrame) Indicator matrix from triple barrier events
    :param sample_length: (int) Length of bootstrapped sample
    :param warmup_samples: (list) List of previously drawn samples
    :param compare: (boolean) Flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
    :param verbose: (boolean) Flag to print updated probabilities on each step
    :param random_state: (np.random.RandomState) Random state
    :return: (array) Bootstrapped samples indexes
    """

    pass
