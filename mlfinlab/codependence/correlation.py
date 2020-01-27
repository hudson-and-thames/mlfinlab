"""
Correlation based distances and various modifications (angular, absolute, squared) described in Cornell lecture notes:
Codependence: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist


# pylint: disable=invalid-name


def angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns angular distance between two vectors. Angular distance is a slight modification of correlation which
    satisfies metric conditions.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :return: (float) angular distance.
    """
    corr_coef = np.corrcoef(x, y)[0][1]
    return np.sqrt(0.5 * (1 - corr_coef))


def absolute_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns a modification of angular distance where absolute value of correlation coefficient is used.

    :param x: (np.array) x vector
    :param y: (np.array) y vector
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


def distance_correlation(x: np.array, y: np.array) -> float:
    """
    Distance correlation captures both linear and non-linear dependencies.
    Distance correlation coefficient is described in https://en.wikipedia.org/wiki/Distance_correlation

    :param x: (np.array) X vector
    :param y: (np.array) Y vector
    :return: (float) distance correlation coefficient
    """

    x = x[:, None]
    y = y[:, None]

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    a = squareform(pdist(x))
    b = squareform(pdist(y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    d_cov_xx = (A * A).sum() / (x.shape[0] ** 2)
    d_cov_xy = (A * B).sum() / (x.shape[0] ** 2)
    d_cov_yy = (B * B).sum() / (x.shape[0] ** 2)

    coef = np.sqrt(d_cov_xy) / np.sqrt(np.sqrt(d_cov_xx) * np.sqrt(d_cov_yy))

    return coef
