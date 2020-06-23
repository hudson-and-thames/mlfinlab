# pylint: disable=bare-except
"""
Implements Signals.
"""

import warnings
import numpy as np


def calc_zscore(data):
    """
    Calculates the z-score for the given data.

    :param data: (np.array) Data for z-score calculation.
    :return: (np.array) Z-score of the given data.
    """
    # Suppress divide by zero warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Change nan to 0.
        res = np.nan_to_num((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    return res


def calc_ou_process(data):
    """
    Calculates the s-score, derived from the Ornstein-Uhlenbeck process, for the given data.

    :param data: (np.array) Data for s-score calculation.
    :return: (tuple) (np.array) S-score of the given data, (np.array) Mean-reverting time.
    """
    # Create resulting np.array.
    ou_process = np.zeros(data.shape)
    mr_speed = np.zeros(data.shape)

    # Iterate through all the columns.
    for i in range(data.shape[1]):
        # Fill in each column.
        ou_process[:, [i]], mr_speed[:, [i]] = _s_score(data[:, i])
    return ou_process, mr_speed


def _s_score(_data):
    """
    Helper function to loop each column for s_score.

    :param _data: (np.array) Data for s-score calculation.
    :return: (tuple) (np.array) S-score of the given data, (np.array) Time scale for mean reversion.
    """
    _data = _data.reshape((_data.size, 1))
    # Shift x down 1.
    data_x = _data[:-1]

    # Add constant.
    data_x = _add_constant(data_x.reshape((data_x.size, 1)))

    # Shift y up 1.
    data_y = _data[1:]

    # Calculate beta.
    beta = _linear_regression(data_x, data_y)

    # Calculate residuals.
    resid = data_y - data_x.dot(beta)

    # Set variables.
    aa = beta[-1]
    bb = beta[0]
    zeta = np.var(resid, axis=0)
    mm = aa / (1 - bb)
    var_eq = np.sqrt(zeta / (1 - bb ** 2))

    # Suppress log warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = -np.log(np.abs(bb)) * 252

    # Set signal and suppress divide by zero warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signal = (_data - mm) / var_eq
    return signal, 1 / kappa


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


def _add_constant(returns):
    """
    Adds a constant of 1 on the right side of the given returns.

    :param returns: (np.array) Log returns for a given time series.
    :return: (np.array) Log returns with an appended column of 1 on the right.
    """
    #  Adds a column of 1 on the right side of the given array.
    return np.hstack((returns, np.ones((returns.shape[0], 1))))
