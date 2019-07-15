"""
Utilities functions for data operation
"""
from __future__ import division
import math
import numpy as np


def calculate_entropy(target):
    """ Calculate the entropy of label array target """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(target)
    entropy = 0
    for label in unique_labels:
        count = len(target[target == label])
        prob = count / len(target)
        entropy += -prob * log2(prob)
    return entropy


def calculate_gini_index(target):
    """ Calculate gini index for label array y """
    unique_labels = np.unique(target)
    gini_index = 0
    for label in unique_labels:
        count = len(target[target == label])
        prob = count / len(target)
        gini_index += prob**2
    gini_index = 1 - gini_index
    return gini_index


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_variance(feat):
    """ Return the variance of the features in dataset feat """
    mean = np.ones(np.shape(feat)) * feat.mean(0)
    n_samples = np.shape(feat)[0]
    variance = (1 / n_samples) * np.diag((feat - mean).T.dot(feat - mean))

    return variance


def calculate_std_dev(feat):
    """ Calculate the standard deviations of the features in dataset feat """
    std_dev = np.sqrt(calculate_variance(feat))
    return std_dev


def euclidean_distance(coord1, coord2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in enumerate(coord1):
        distance += pow((coord1[i[0]] - coord2[i[0]]), 2)
    return math.sqrt(distance)


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_covariance_matrix(feat, trg=None):
    """ Calculate the covariance matrix for the dataset feat """
    if trg is None:
        trg = feat
    n_samples = np.shape(feat)[0]
    covariance_matrix = (1 / (n_samples-1)) * (feat - feat.mean(axis=0)).T.dot(trg - trg.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


def calculate_correlation_matrix(feat, trg=None):
    """ Calculate the correlation matrix for the dataset feat """
    if trg is None:
        trg = feat
    n_samples = np.shape(feat)[0]
    covariance = (1 / n_samples) * (feat - feat.mean(0)).T.dot(trg - trg.mean(0))
    target_std = np.expand_dims(calculate_std_dev(feat), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(trg), 1)
    correlation_matrix = np.divide(covariance, target_std.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)
