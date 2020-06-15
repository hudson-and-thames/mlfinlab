"""
Calculate regression.
"""

import numpy as np
import pandas as pd


def calc_all_regression(data_x, data_y):
    """
    Calculate slope, intercept, spread, and z-score for time series data_x and data_y.

    :param data_x: (pd.Series) Time series x.
    :param data_y: (pd.Series) Time series y.
    :return: (pd.DataFrame) DataFrame of given data_x, data_y, slope, intercept, spread, and z-score.
    """
    # Convert data_x to np.array.
    np_x = np.array(data_x)

    # Add 1 to all values to account for intercept.
    np_x = np.vstack((np_x, np.ones(np_x.shape))).T

    # Convert data_y to np.array.
    np_y = np.array([data_y]).T

    # Calculate beta, the slope and intercept.
    beta = np.linalg.inv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y)

    # Calculate spread.
    spread = np_y - np_x.dot(beta)

    # Calculate standard deviation.
    st_dev = np.std(spread)

    # Calculate z-score.
    z_score = spread - np.mean(spread) / st_dev

    # Column of slope and intercept.
    beta = np.repeat(beta, np_x.shape[0], axis=1).T

    # Stack all values.
    res = np.hstack((np_x[:, [0]], np_y, beta, spread, z_score))

    # Columns name.
    col_name = ['data_x', 'data_y', 'slope', 'intercept', 'spread', 'z_score']
    return pd.DataFrame(res, columns=col_name, index=data_x.index)


def calc_rolling_regression(data_x, data_y, window):
    """
    Calculate slope, intercept, spread, and z-score for time series data_x and data_y for a rolled
    window.

    :param data_x: (pd.Series) Time series x.
    :param data_y: (pd.Series) Time series y.
    :param window: (int) Number of rolled windows.
    :return: (pd.DataFrame) DataFrame of given data_x, data_y, slope, intercept, spread, and z-score
            for each rolled window.
    """
    # Convert data_x to np.array.
    np_x = np.array(data_x)

    # Add 1 to all values to account for intercept.
    np_x = np.vstack((np_x, np.ones(np_x.shape))).T

    # Convert data_y to np.array.
    np_y = np.array([data_y])

    # Combined data.
    data = np.hstack((np_x, np_y.T))

    # Rolled data.
    data = _rolling_window(data, window)

    # Initialize result.
    res = np.zeros((np_x.shape[0], 6))

    for i in range(data.shape[0]):
        res[i + window - 1] = _calc_rolling_params(data[i])

    # Set nan for beginning windows.
    res[:window - 1] = np.nan

    # Columns name.
    col_name = ['data_x', 'data_y', 'slope', 'intercept', 'spread', 'z_score']
    return pd.DataFrame(res, index=data_x.index, columns=col_name)


def _calc_rolling_params(data):
    # pylint: disable=bare-except
    """
    Helper function to calculate rolling parameters

    :param data: (np.array) Rolling window of original data.
    :return: (np.array) Data_x, data_y, slope, intercept, spread, and z-score.
    """
    # Split data to np_x
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

    # Calculate standard deviation.
    st_dev = np.std(spread)

    # Calculate last entry's z-score.
    z_score = (spread[-1] - np.mean(spread)) / st_dev

    res = np.array([np_x[-1][0], np_y[-1], beta[0], beta[1], spread[-1], z_score])

    return res


def _rolling_window(data, window):
    """
    Helper function to generate a rolled window.

    :param data: (np.array) Original data given by user.
    :param window: (int) Number of rolled window.
    :return: (np.array) All generated windows.
    """
    shape = (data.shape[0] - window + 1, window) + data.shape[1:]
    strides = (data.strides[0],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
