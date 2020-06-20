# pylint: disable=bare-except
"""
Calculate pairs regression.
"""

import numpy as np
import pandas as pd


def calc_all_pairs_regression(data_x, data_y):
    """
    Calculate data_x log returns, data_y log returns, beta, constant, ret_spread, cum_resid,
    and z-score for time series of data_x and data_y.

    :param data_x: (pd.Series) Time series of price of x (DO NOT adjust for log).
    :param data_y: (pd.Series) Time series of price of y (DO NOT adjust for log).
    :return: (pd.DataFrame) DataFrame of given data_x, data_y, logret_x, logret_y, beta, constant,
        ret_spread, cum_resid, and z-score.
    """
    # Change to log returns and conver to np.array.
    np_x = np.array(np.log(data_x).diff().fillna(0))
    np_y = np.array([np.log(data_y).diff().fillna(0)])

    # Add 1 to all values to account for intercept.
    np_x = np.vstack((np_x, np.ones(np_x.shape))).T

    # Calculate beta, the slope and intercept.
    try:
        beta = np.linalg.inv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y.T)
    except:
        beta = np.linalg.pinv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y.T)

    # Calculate spread of returns.
    spread = np_y.T - np_x.dot(beta)

    # Calculate cumulative sum of spread of returns.
    cum_resid = spread.cumsum()

    # Calculate z-score.
    z_score = (cum_resid - np.mean(cum_resid)) / np.std(cum_resid)

    # Column of slope and intercept.
    beta = np.repeat(beta, np_x.shape[0], axis=1).T

    # Stack all values.
    res = np.hstack((np.array([data_x]).T, np.array([data_y]).T, np_x[:, [0]], np_y.T, beta, spread,
                     np.vstack((cum_resid, z_score)).T))

    # Columns name.
    col_name = [data_x.name, data_y.name, 'logret_x', 'logret_y', 'beta', 'constant', 'ret_spread',
                'cum_resid', 'z_score']
    return pd.DataFrame(res, columns=col_name, index=data_x.index)


def calc_rolling_pairs_regression(data_x, data_y, window):
    """
    Calculate data_x log returns, data_y log returns, beta, constant, ret_spread, cum_resid,
    and z-score for time series of data_x and data_y with the given window.

    :param data_x: (pd.Series) Time series of price of x (DO NOT adjust for log).
    :param data_y: (pd.Series) Time series of price of y (DO NOT adjust for log).
    :param window: (int) Number of rolling windows.
    :return: (pd.DataFrame) DataFrame of given data_x, data_y, logret_x, logret_y, beta, constant,
        ret_spread, and z-score.
    """
    # Change to log returns and conver to np.array.
    np_x = np.array(np.log(data_x).diff().fillna(0))
    np_y = np.array([np.log(data_y).diff().fillna(0)])

    # Add 1 to all values to account for intercept.
    np_x = np.vstack((np_x, np.ones(np_x.shape))).T

    # Combined data.
    data = np.hstack((np_x, np_y.T))

    # Rolled data.
    data = _rolling_window(data, window)

    # Initialize result.
    res = np.zeros((np_x.shape[0], 5))

    # Fill in the array.
    for i in range(data.shape[0]):
        res[i + window - 1] = _calc_rolling_reg_params(data[i])

    # Set nan for beginning windows.
    res[:window - 1] = np.nan

    # Stack original data and log returns with the results.
    res = np.hstack((np.array([data_x]).T, np.array([data_y]).T, np_x[:, [0]], np_y.T, res))
    # Columns name.
    col_name = [data_x.name, data_y.name, 'logret_x', 'logret_y', 'beta', 'constant', 'ret_spread',
                'cum_resid', 'z_score']
    return pd.DataFrame(res, index=data_x.index, columns=col_name)


def _calc_rolling_reg_params(data):
    """
    Helper function to calculate rolling regression parameters.

    :param data: (np.array) Rolling window of original data.
    :return: (np.array) Data_x, data_y, beta, constant, spread, cum_resid, and z-score.
    """
    # Split data to np_x.
    np_x = data[:, [0, 1]]

    # Split data to np_y.
    np_y = data[:, [2]]

    # Calculate beta, the slope and intercept.
    try:
        beta = np.linalg.inv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y)
    except:
        beta = np.linalg.pinv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y)

    # Calculate spread.
    spread = np_y - np_x.dot(beta)

    # Calculate cumulative sum of spread of returns.
    cum_resid = spread.cumsum()

    # Calculate z-score.
    z_score = (cum_resid[-1] - np.mean(cum_resid)) / np.std(cum_resid)

    # Separate the resulting array.
    res = np.array([beta[0][0], beta[1][0], spread[-1][0], cum_resid[-1], z_score])

    return res


def _rolling_window(data, window):
    """
    Helper function to generate rolling windows.

    :param data: (np.array) Original data given by user.
    :param window: (int) Number of rolling window.
    :return: (np.array) All generated windows.
    """
    shape = (data.shape[0] - window + 1, window) + data.shape[1:]
    strides = (data.strides[0],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
