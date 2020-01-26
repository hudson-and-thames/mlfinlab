"""
Correlation based distances (angular, absolute, squared) described in
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""

import numpy as np


def angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns angular distance between two vectors. Angular distance is a slight modification of correlation which
    satisfies metric conditions.
    :param x: (np.array) X vector
    :param y: (np.array) Y vector
    :return: (float) angular distance
    """
    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - corr_coef))


def absolute_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns a modification of angular distance where absolute value of correlation coefficient is used.
    :param x: (np.array) X vector
    :param y: (np.array) Y vector
    :return: (float) absolute angular distance
    """

    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - abs(corr_coef)))


def squared_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns a modification of angular distance where square of correlation coefficient is used.
    :param x: (np.array) X vector
    :param y: (np.array) Y vector
    :return: (float) squared angular distance
    """

    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - corr_coef ** 2))
