"""
Chapter 3.2 Fixed-Time Horizon Method
Described in "Classification-based Financial Markets Prediction using Deep Neural Networks" Dixon et al. (2016)
"""

import numpy as np
import pandas as pd


def fixed_time_horizon(close, threshold, lookfwd=1, standardized=None):
    """
    Fixed-Time Horizon Labelling Method
    Returns 1 if return at h-th bar after t_0 is greater than the threshold, -1 if less, and 0 if in between

    :param close: (pd.Series) of Close prices
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, it is labelled as 1 or -1.
                    If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series
    :param lookfwd: : (int) Number of ticks to look forward when calculating future return rate
    :param standardized: (pd.DataFrame, or DataFrame-like) of (mean, stdev) of returns corresponding to days in close
    :return: (pd.Series) Series of -1, 0, or 1 denoting whether return is under/between/greater than the threshold
    """
    # Calculate forward price with
    fwd = close.pct_change(periods=lookfwd).shift(-lookfwd)

    # Adjust by mean and stdev, if provided
    if standardized is not None:
        standardize = pd.DataFrame(standardized, columns=['mean', 'sd'], index=fwd.index)
        fwd -= standardize['mean']
        fwd /= standardize['sd']

    # Conditions for 1, 0, -1
    conditions = [fwd > threshold, (fwd <= threshold) & (fwd >= -threshold), fwd < -threshold]
    choices = [1, 0, -1]
    labels = np.select(conditions, choices, default=np.nan)

    return labels
