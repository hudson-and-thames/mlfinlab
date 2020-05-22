"""
Chapter 3.2 Fixed-Time Horizon Method, in Advances in Financial Machine Learning, by M. L. de Prado

Work "Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
labeling data this way can be used in training deep neural networks to predict price movements.
"""

import warnings
import numpy as np
import pandas as pd


def fixed_time_horizon(close, threshold=0, look_forward=1, standardized=False, window=None):
    """
    Fixed-Time Horizon Labelling Method

    Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.

    Returns 1 if return for a period is greater than the threshold, -1 if less, and 0 if in between. If no threshold is
    provided then it will simply take the sign of the return.

    :param close: (pd.Series) Close prices over fixed horizons (usually time bars, but can be any format as long as
                    index is timestamps) for a stock ticker.
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, it is labelled as 1 or -1.
                    If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series. If threshold is
                    a series, threshold.index must match close.index. If threshold is negative, then the directionality
                    of the labels will be reversed. If no threshold is given, then the sign of the observation is
                    returned.
    :param look_forward: (int) Number of ticks to look forward when calculating future return rate. (1 by default)
                        If n is the numerical value of look_forward, the last n observations will return a label of NaN
                        due to lack of data to calculate the forward return in those cases.
    :param standardized: (bool) Whether returns are scaled by mean and standard deviation.
    :param window: (int) If standardized is True, the rolling window period for calculating the mean and standard
                    deviation of returns.
    :return: (pd.Series) -1, 0, or 1 denoting whether return for each tick is under/between/greater than the threshold.
                    The final look_forward number of observations will be labeled np.nan. Index is same as index of
                    close.
    """
    # Calculate returns
    forward_return = close.pct_change(periods=look_forward).shift(-look_forward)

    # Warning if look_forward is greater than the length of the series.
    if look_forward >= len(forward_return):
        warnings.warn('look_forward period is greater than the length of the Series. All labels will be NaN.',
                      UserWarning)

    # Adjust by mean and stdev, if desired. Assert that window must exist if standardization is on. Warning if window is
    # too large.
    if standardized:
        # Error handling
        assert isinstance(window, int), "when standardized is True, window must be int"
        if window >= len(forward_return):
            warnings.warn('window is greater than the length of the Series. All labels will be NaN.', UserWarning)

        # Apply standardisation
        mean = forward_return.rolling(window=window).mean()
        stdev = forward_return.rolling(window=window).std()
        forward_return -= mean
        forward_return /= stdev

    # Apply labeling otherwise
    conditions = [forward_return > threshold, (forward_return <= threshold) & (forward_return >= -threshold),
                  forward_return < -threshold]
    choices = [1, 0, -1]
    labels = np.select(conditions, choices, default=np.nan)

    return pd.Series(labels, index=close.index)
