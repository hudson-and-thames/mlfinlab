"""
Implements Signals.
"""

import numpy as np
import pandas as pd
import warnings


def z_score(data):
    """
    Calculates the z-score for the given data.

    :param data: (np.array) Data for z-score calculation.
    :return: (np.array) Z-score of the given data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.nan_to_num((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    return res


def s_score(data):
    """
    Calculates the s-score for the given data.

    :param data: (np.array) Data for s-score calculation.
    :return: (np.array) S-score of the given data.
    """
    return


def _linear_regression(data_x, data_y):
    """
    Calculates the parameter vector using matrix multiplication.

    :param data_x: (np.array) Time series of log returns of x.
    :param data_y: (np.array) Time series of log returns of y.
    :return: (np.array) Parameter vector.
    """
    try:
        beta = np.linalg.inv(data_x.T.dot(data_x)).dot(data_x.T).dot(data_y)
    except:
        beta = np.linalg.pinv(data_x.T.dot(data_x)).dot(data_x.T).dot(data_y)
    return beta
