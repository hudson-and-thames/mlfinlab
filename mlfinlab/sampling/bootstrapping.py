"""
Logic regarding sequential bootstrapping from chapter 4.
"""

import pandas as pd
import numpy as np
from numba import jit, prange


def get_ind_matrix(samples_info_sets, price_bars):
    """
    Snippet 4.3, page 65, Build an Indicator Matrix
    Get indicator matrix. The book implementation uses bar_index as input, however there is no explanation
    how to form it. We decided that using triple_barrier_events and price bars by analogy with concurrency
    is the best option.

    :param samples_info_sets: (pd.Series): triple barrier events(t1) from labeling.get_events
    :param price_bars: (pd.DataFrame): price bars which were used to form triple barrier events
    :return: (np.array) indicator binary matrix indicating what (price) bars influence the label for each observation
    """
    if bool(samples_info_sets.isnull().values.any()) is True or bool(
            samples_info_sets.index.isnull().any()) is True:
        raise ValueError('NaN values in triple_barrier_events, delete nans')

    triple_barrier_events = pd.DataFrame(samples_info_sets)  # Convert Series to DataFrame

    # Take only period covered in triple_barrier_events
    trimmed_price_bars_index = price_bars[(price_bars.index >= triple_barrier_events.index.min()) &
                                          (price_bars.index <= triple_barrier_events.t1.max())].index

    label_endtime = triple_barrier_events.t1
    bar_index = list(triple_barrier_events.index)  # Generate index for indicator matrix from t1 and index
    bar_index.extend(triple_barrier_events.t1)
    bar_index.extend(trimmed_price_bars_index)  # Add price bars index
    bar_index = sorted(list(set(bar_index)))  # Drop duplicates and sort

    # Get sorted timestamps with index in sorted array
    sorted_timestamps = dict(zip(sorted(bar_index), range(len(bar_index))))

    tokenized_endtimes = np.column_stack((label_endtime.index.map(sorted_timestamps), label_endtime.map(
        sorted_timestamps).values))  # Create array of arrays: [label_index_position, label_endtime_position]

    ind_mat = np.zeros((len(bar_index), len(label_endtime)))  # Init indicator matrix
    for sample_num, label_array in enumerate(tokenized_endtimes):
        label_index = label_array[0]
        label_endtime = label_array[1]
        ones_array = np.ones(
            (1, label_endtime - label_index + 1))  # Ones array which corresponds to number of 1 to insert
        ind_mat[label_index:label_endtime + 1, sample_num] = ones_array
    return ind_mat


def get_ind_mat_average_uniqueness(ind_mat):
    """
    Snippet 4.4. page 65, Compute Average Uniqueness
    Average uniqueness from indicator matrix

    :param ind_mat: (np.matrix) indicator binary matrix
    :return: (float) average uniqueness
    """
    concurrency = ind_mat.sum(axis=1)
    uniqueness = ind_mat.T / concurrency

    avg_uniqueness = uniqueness[uniqueness > 0].mean()

    return avg_uniqueness


def get_ind_mat_label_uniqueness(ind_mat):
    """
    An adaption of Snippet 4.4. page 65, which returns the indicator matrix element uniqueness.

    :param ind_mat: (np.matrix) indicator binary matrix
    :return: (np.matrix) element uniqueness
    """
    concurrency = ind_mat.sum(axis=1)
    uniqueness = ind_mat.T / concurrency

    return uniqueness


@jit(parallel=True, nopython=True)
def _bootstrap_loop_run(ind_mat, prev_concurrency):  # pragma: no cover
    """
    Part of Sequential Bootstrapping for-loop. Using previously accumulated concurrency array, loops through all samples
    and generates averages uniqueness array of label based on previously accumulated concurrency
    :param ind_mat (np.array): indicator matrix from get_ind_matrix function
    :param prev_concurrency (np.array): accumulated concurrency from previous iterations of sequential bootstrapping
    :return: (np.array): label average uniqueness based on prev_concurrency
    """
    avg_unique = np.zeros(ind_mat.shape[1])  # Array of label uniqueness

    for i in prange(ind_mat.shape[1]):  # pylint: disable=not-an-iterable
        prev_average_uniqueness = 0
        number_of_elements = 0
        reduced_mat = ind_mat[:, i]
        for j in range(len(reduced_mat)):  # pylint: disable=consider-using-enumerate
            if reduced_mat[j] > 0:
                new_el = reduced_mat[j] / (reduced_mat[j] + prev_concurrency[j])
                average_uniqueness = (prev_average_uniqueness * number_of_elements + new_el) / (number_of_elements + 1)
                number_of_elements += 1
                prev_average_uniqueness = average_uniqueness
        avg_unique[i] = average_uniqueness
    return avg_unique


def seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None, compare=False, verbose=False,
                  random_state=np.random.RandomState()):
    """
    Snippet 4.5, Snippet 4.6, page 65, Return Sample from Sequential Bootstrap
    Generate a sample via sequential bootstrap.
    Note: Moved from pd.DataFrame to np.matrix for performance increase

    :param ind_mat: (data frame) indicator matrix from triple barrier events
    :param sample_length: (int) Length of bootstrapped sample
    :param warmup_samples: (list) list of previously drawn samples
    :param compare: (boolean) flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
    :param verbose: (boolean) flag to print updated probabilities on each step
    :param random_state: (np.random.RandomState) random state
    :return: (array) of bootstrapped samples indexes
    """

    if sample_length is None:
        sample_length = ind_mat.shape[1]

    if warmup_samples is None:
        warmup_samples = []

    phi = []  # Bootstrapped samples
    prev_concurrency = np.zeros(ind_mat.shape[0])  # Init with zeros (phi is empty)
    while len(phi) < sample_length:
        avg_unique = _bootstrap_loop_run(ind_mat, prev_concurrency)
        prob = avg_unique / sum(avg_unique)  # Draw prob
        try:
            choice = warmup_samples.pop(0)  # It would get samples from warmup until it is empty
            # If it is empty from the beginning it would get samples based on prob from the first iteration
        except IndexError:
            choice = random_state.choice(range(ind_mat.shape[1]), p=prob)
        phi += [choice]
        prev_concurrency += ind_mat[:, choice]  # Add recorded label array from ind_mat
        if verbose is True:
            print(prob)

    if compare is True:
        standard_indx = np.random.choice(ind_mat.shape[1], size=sample_length)
        standard_unq = get_ind_mat_average_uniqueness(ind_mat[:, standard_indx])
        sequential_unq = get_ind_mat_average_uniqueness(ind_mat[:, phi])
        print('Standard uniqueness: {}\nSequential uniqueness: {}'.format(standard_unq, sequential_unq))

    return phi
