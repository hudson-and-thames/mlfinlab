"""
Utilities functions for data manipulation
"""
from __future__ import division
from itertools import combinations_with_replacement
import numpy as np


def shuffle_data(feat, trg, seed=None):
    """ Random shuffle of the samples in feat and trg """
    if seed:
        np.random.seed(seed)
    idx = np.arange(feat.shape[0])
    np.random.shuffle(idx)
    return feat[idx], trg[idx]


def batch_iterator(feat, trg=None, batch_size=64):
    """ Simple batch generator """
    n_samples = feat.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if trg is not None:
            yield feat[begin:end], trg[begin:end]
        else:
            yield feat[begin:end]


def divide_on_feature(feat, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, (int, float)):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    feat_1 = np.array([sample for sample in feat if split_func(sample)])
    feat_2 = np.array([sample for sample in feat if not split_func(sample)])

    return np.array([feat_1, feat_2])


def polynomial_features(feat, degree):
    """
    Create polynomial features.
    :param feat:
    :param degree:
    :return:
    """
    n_samples, n_features = np.shape(feat)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()
    n_output_features = len(combinations)
    feat_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        feat_new[:, i] = np.prod(feat[:, index_combs], axis=1)

    return feat_new


def get_random_subsets(feat, trg, n_subsets, replacements=True):
    """ Return random subsets (with replacements) of the data """
    n_samples = np.shape(feat)[0]
    # Concatenate feat and trg and do a random shuffle
    feat_trg = np.concatenate((feat, trg.reshape((1, len(trg))).T), axis=1)
    np.random.shuffle(feat_trg)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples      # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        feat = feat_trg[idx][:, :-1]
        trg = feat_trg[idx][:, -1]
        subsets.append([feat, trg])
    return subsets


def normalize(feat, axis=-1, order=2):
    """ Normalize the dataset feat """
    norm_l2 = np.atleast_1d(np.linalg.norm(feat, order, axis))
    norm_l2[norm_l2 == 0] = 1
    return feat / np.expand_dims(norm_l2, axis)


def standardize(feat):
    """ Standardize the dataset feat """
    feat_std = feat
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    for col in range(np.shape(feat)[1]):
        if std[col]:
            feat_std[:, col] = (feat_std[:, col] - mean[col]) / std[col]
    # feat_std = (feat - feat.mean(axis=0)) / feat.std(axis=0)
    return feat_std


def train_test_split(feat, trg, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        feat, trg = shuffle_data(feat, trg, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(trg) - int(len(trg) // (1 / test_size))
    train_feat, test_feat = feat[:split_i], feat[split_i:]
    train_trg, test_trg = trg[:split_i], trg[split_i:]

    return train_feat, test_feat, train_trg, test_trg


def k_fold_cross_validation_sets(feat, trg, k, shuffle=True):
    """ Split the data into k sets of training / test data """
    if shuffle:
        feat, trg = shuffle_data(feat, trg)

    n_samples = len(trg)
    left_overs = {}
    n_left_overs = (n_samples % k)
    if n_left_overs != 0:
        left_overs["feat"] = feat[-n_left_overs:]
        left_overs["trg"] = trg[-n_left_overs:]
        feat = feat[:-n_left_overs]
        trg = trg[:-n_left_overs]

    feat_split = np.split(feat, k)
    y_split = np.split(trg, k)
    sets = []
    for i in range(k):
        test_feat, test_trg = feat_split[i], y_split[i]
        train_feat = np.concatenate(feat_split[:i] + feat_split[i + 1:], axis=0)
        train_trg = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
        sets.append([train_feat, test_feat, train_trg, test_trg])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["feat"], axis=0)
        np.append(sets[-1][2], left_overs["trg"], axis=0)

    return np.array(sets)


def to_categorical(data, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(data) + 1
    one_hot = np.zeros((data.shape[0], n_col))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot


def to_nominal(data):
    """ Conversion from one-hot encoding to nominal """
    return np.argmax(data, axis=1)


def make_diagonal(data):
    """ Converts a vector into an diagonal matrix """
    matrix = np.zeros((len(data), len(data)))
    for i in range(len(matrix[0])):
        matrix[i, i] = data[i]
    return matrix
