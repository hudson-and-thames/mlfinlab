"""
Logic regarding sequential bootstrapping from chapter 4.
"""

import numpy as np
import pandas as pd


def get_ind_matrix(bar_index, label_endtime):
    """
    Snippet 4.3, page 64, Build an Indicator Matrix
    Get indicator matrix

    :param bar_index: (pd.Series): Index of bars
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :return: (pd.DataFrame) indicator binary matrix indicating what (price) bars influence the label for each observation
    """
    ind_mat = pd.DataFrame(0, index=bar_index, columns=range(label_endtime.shape[0]))  # Zero indicator matrix
    assert bool(ind_mat.isnull().values.any()) is False and bool(
        ind_mat.index.isnull().any()) is False, 'NaN values in bar_index or label_endtime, delete nans'

    # loop through t1 events and index, put 1 where we have an overlap across index
    for i, (t_0, t_1) in enumerate(label_endtime.iteritems()):
        ind_mat.loc[t_0:t_1, i] = 1
    return ind_mat


def get_ind_mat_average_uniqueness(ind_mat):
    """
    Snippet 4.4. page 65, Compute Average Uniqueness
    Average uniqueness from indicator matrix

    :param ind_mat: (np.matrix) indicator binary matrix
    :return: (np.matrix) matrix with label uniqueness
    """
    conc = ind_mat.sum(axis=1)  # concurrency
    average = ind_mat / conc[:, None]
    return average.T


def seq_bootstrap(triple_barrier_events, sample_length=None, compare=False, random_state=None):
    """
    Snippet 4.5, Snippet 4.6, page 65, Return Sample from Sequential Bootstrap
    Generate a sample via sequential bootstrap.
    Note: Moved from pd.DataFrame to np.matrix for performance increase

    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param random_state: (np.mtrand.RandomState) random state object for generating random numbers
    :param sample_length: (int) Length of bootstrapped sample
    :param compare: (boolean) flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
    :return: (array) of bootstrapped samples indexes
    """
    assert bool(triple_barrier_events.isnull().values.any()) is False and bool(
        triple_barrier_events.index.isnull().any()) is False, 'NaN values in triple_barrier_events, delete nans'

    if random_state is None:
        random_state = np.random.mtrand.RandomState()

    label_endtime = triple_barrier_events.t1

    bar_index = list(triple_barrier_events.index)  # Generate index for indicator matrix from t1 and index
    bar_index.extend(triple_barrier_events.t1)
    bar_index = sorted(list(set(bar_index)))  # Drop duplicates and sort

    ind_mat = get_ind_matrix(bar_index, label_endtime)

    if sample_length is None:
        sample_length = ind_mat.shape[1]

    phi = []
    while len(phi) < sample_length:
        avg_unique = np.array([])
        for i in ind_mat:  # TODO: For performance increase, this can be parallelized
            ind_mat_reduced = ind_mat[phi + [i]]  # Reduce ind_mat (append i label as last column)
            # Get i label uniqueness vector (which corresponds to the last column of get_ind_mat_average_uniqueness)
            label_uniqueness = get_ind_mat_average_uniqueness(ind_mat_reduced.values)[-1]
            label_av_uniqueness = label_uniqueness[label_uniqueness > 0].mean()  # Get average label uniqueness
            avg_unique = np.append(avg_unique, label_av_uniqueness)
        prob = avg_unique / avg_unique.sum()  # Draw prob
        phi += [random_state.choice(ind_mat.columns, p=prob)]

    if compare is True:
        standard_indx = np.random.choice(ind_mat.columns, size=sample_length)
        standart_unq = get_ind_mat_average_uniqueness(ind_mat[standard_indx].values)
        standard_unq_mean = standart_unq[standart_unq > 0].mean()

        sequential_unq = get_ind_mat_average_uniqueness(ind_mat[phi].values)
        sequential_unq_mean = sequential_unq[sequential_unq > 0].mean()
        print('Standard uniqueness: {}\nSequential uniqueness: {}'.format(standard_unq_mean, sequential_unq_mean))

    return phi
