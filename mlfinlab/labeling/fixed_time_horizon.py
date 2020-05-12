"""
Chapter 3.2 Fixed-Time Horizon Method
Described in "Classification-based Financial Markets Prediction using Deep Neural Networks" Dixon et al. (2016)
"""

import numpy as np
import pandas as pd


def get_forward_return(close, lookfwd):
    """
    Gets forward return of a series of close prices, given a specified number of ticks to look forward
    :param close: (series) of Close prices
    :param lookfwd: (int) Number of ticks to look forward when calculating forward return rate
    :return: (series) of Forward returns
    """
    # Daily returns
    daily_ret = close.pct_change(periods=lookfwd)

    # "Forward" returns h time intervals in the future, to be compared against the threshold
    forward_ret = pd.Series(list(daily_ret)[lookfwd:] + [float("NaN")] * lookfwd, index=close.index)

    return forward_ret


def standardize(forward_ret, standardized):
    """
    Applies standardization a pd.Series of returns for a stock when the market return and standard deviation are
    known
    :param forward_ret: (series) of Forward returns
    :param standardized: (list) of Tuples (mean, stdev) of returns corresponding to each day in forward_ret
    :return: (series) of Standardized returns
    """
    mean = pd.Series([i for i, j in standardized], index=forward_ret.index)
    stdev = pd.Series([j for i, j in standardized], index=forward_ret.index)
    forward_ret_adj = forward_ret.sub(mean)
    forward_ret_adj = forward_ret_adj.div(stdev)

    return forward_ret_adj


def fixed_time_horizon(close, threshold, lookfwd=1, standardized=None):
    """
    Fixed-Time Horizon Labelling Method
    Returns 1 if return at h-th bar after t_0 is greater than the threshold, -1 if less, and 0 if in between

    :param close: (series) of Close prices
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, it is labelled as 1 or -1.
                    If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series
    :param lookfwd: : (int) Number of ticks to look forward when calculating future return rate
    :param standardized: (list) of Tuples (mean, stdev) of returns corresponding to each day in close. If not None, the
                    forward returns are for each day are adjusted by the mean and stdev
    :return: (series) Series of -1, 0, or 1 denoting whether return is under/between/greater than the threshold
    """
    # Forward returns
    forward_ret = get_forward_return(close, lookfwd)

    # If standardization is applied, adjust forward_ret by mean and stdev
    if standardized is not None:
        forward_ret = standardize(forward_ret, standardized)

    # Compare forward return with the threshold, and returns -1, 0, 1 for lower than/between/greater than threshold
    if isinstance(threshold, (float, int)):
        labels = forward_ret.apply(
            lambda row: 1 if row > threshold else (
                0 if threshold > row > -threshold else (-1 if row < -threshold else np.nan)))

    elif isinstance(threshold, pd.Series):
        labels = []
        ret_to_threshold = [(i, j) for (i, j) in zip(forward_ret.tolist(), threshold.to_list())]
        for ret, thrsh in ret_to_threshold:
            if ret > thrsh:
                labels.append(1)
            elif -thrsh < ret < thrsh:
                labels.append(0)
            elif ret < -thrsh:
                labels.append(-1)
            else:
                labels.append(np.nan)
        labels = pd.Series(labels, index=forward_ret.index)

    else:
        raise ValueError('threshold is neither float nor pd.Series!')

    return labels
