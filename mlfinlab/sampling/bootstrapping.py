"""
Logic regarding sequential bootstrapping from chapter 4.
"""

import numpy as np
from numba import jit, prange


def get_ind_matrix(triple_barrier_events):
    """
    Snippet 4.3, page 64, Build an Indicator Matrix
    Get indicator matrix
    :param triple_barrier_events: (pd.DataFrame): triple barrier events from labeling.get_events
    :return: (np.array) indicator binary matrix indicating what (price) bars influence the label for each observation
    """
    if bool(triple_barrier_events.isnull().values.any()) is True or bool(
            triple_barrier_events.index.isnull().any()) is True:
        raise ValueError('NaN values in triple_barrier_events, delete nans')

    label_endtime = triple_barrier_events.t1
    bar_index = list(triple_barrier_events.index)  # generate index for indicator matrix from t1 and index
    bar_index.extend(triple_barrier_events.t1)
    bar_index = sorted(list(set(bar_index)))  # drop duplicates and sort

    sorted_timestamps = dict(
        zip(sorted(bar_index), range(len(bar_index))))  # get sorted timestamps with index in sorted array

    tokenized_endtimes = np.column_stack((label_endtime.index.map(sorted_timestamps), label_endtime.map(
        sorted_timestamps).values))  # create array of arrays: [label_index_position, label_endtime_position]

    ind_mat = np.zeros((len(bar_index), len(label_endtime)))  # init indicator matrix
    for sample_num, label_array in enumerate(tokenized_endtimes):
        label_index = label_array[0]
        label_endtime = label_array[1]
        ones_array = np.ones(
            (1, label_endtime - label_index + 1))  # ones array which corresponds to number of 1 to insert
        ind_mat[label_index:label_endtime + 1, sample_num] = ones_array
    return ind_mat


def get_ind_mat_average_uniqueness(ind_mat):
    """
    Snippet 4.4. page 65, Compute Average Uniqueness
    Average uniqueness from indicator matrix

    :param ind_mat: (np.matrix) indicator binary matrix
    :return: (np.matrix) matrix with label uniqueness
    """
    conc = ind_mat.sum(axis=1)  # concurrency
    average = ind_mat.T / conc
    return average


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


def seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None, compare=False, verbose=False):
    """
    Snippet 4.5, Snippet 4.6, page 65, Return Sample from Sequential Bootstrap
    Generate a sample via sequential bootstrap.
    Note: Moved from pd.DataFrame to np.matrix for performance increase

    :param ind_mat: (data frame) indicator matrix from triple barrier events
    :param sample_length: (int) Length of bootstrapped sample
    :param warmup_samples: (list) list of previously drawn samples
    :param compare: (boolean) flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
    :param verbose: (boolean) flag to print updated probabilities on each step
    :return: (array) of bootstrapped samples indexes
    """

    random_state = np.random.RandomState()

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
        standard_unq_mean = standard_unq[standard_unq > 0].mean()

        sequential_unq = get_ind_mat_average_uniqueness(ind_mat[:, phi])
        sequential_unq_mean = sequential_unq[sequential_unq > 0].mean()
        print('Standard uniqueness: {}\nSequential uniqueness: {}'.format(standard_unq_mean, sequential_unq_mean))

    return phi
