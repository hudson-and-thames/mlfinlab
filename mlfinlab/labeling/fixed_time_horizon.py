"""
Chapter 3.2 Fixed-Time Horizon Method
Described in "Classification-based Financial Markets Prediction using Deep Neural Networks" Dixon et al. (2016)
"""

import numpy as np
import pandas as pd


def fixed_time_horizon(close, threshold, h=1):
    """
    Fixed-Time Horizon Labelling Method
    Returns 1 if return at h-th bar after t_0 is greater than the threshold, -1 if less, and 0 if in between

    :param close: (series) of close prices
    :param threshold: (float or pd.Series) when the abs(change) is larger than the threshold, it is labelled as 1 or -1.
                    If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series
    :param h: (int) number of ticks to look forward when calculating future return rate
    :return: (series) series of -1, 0, or 1 denoting whether return is under/between/greater than the threshold
    """
    # daily returns
    daily_ret = close.pct_change(periods=h)

    # "forward" returns h time intervals in the future, to be compared against the threshold
    forward_ret = pd.Series(list(daily_ret)[h:] + [float("NaN")] * h, index=close.index)

    # compare forward return with the threshold, and returns -1, 0, 1 for lower than/between/greater than threshold
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
            elif ret < -thrsh:
                labels.append(-1)
            else:
                labels.append(0)
        labels = pd.Series(labels, index=forward_ret.index)

    else:
        raise ValueError('threshold is neither float nor pd.Series!')

    return labels
