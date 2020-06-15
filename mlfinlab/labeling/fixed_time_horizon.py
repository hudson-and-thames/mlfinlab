"""
Chapter 3.2 Fixed-Time Horizon Method, in Advances in Financial Machine Learning, by M. L. de Prado.

Work "Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
labeling data this way can be used in training deep neural networks to predict price movements.
"""

import warnings
import pandas as pd


def fixed_time_horizon(prices, threshold=0, resample_by=None, lag=True, standardized=False, window=None):
    """
    Fixed-Time Horizon Labeling Method.

    Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.

    Returns 1 if return is greater than the threshold, -1 if less, and 0 if in between. If no threshold is
    provided then it will simply take the sign of the return.

    :param prices: (pd.Series or pd.DataFrame) Time-indexed stock prices used to calculate returns.
    :param threshold: (float or pd.Series) When the absolute value of return exceeds the threshold, the observation is
                    labeled with 1 or -1, depending on the sign of the return. If return is less, it's labeled as 0.
                    Can be dynamic if threshold is inputted as a pd.Series, and threshold.index must match prices.index.
                    If resampling is used, the index of threshold must match the index of prices after resampling.
                    If threshold is negative, then the directionality of the labels will be reversed. If no threshold
                    is provided, it is assumed to be 0 and the sign of the return is returned.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :param standardized: (bool) Whether returns are scaled by mean and standard deviation.
    :param window: (int) If standardized is True, the rolling window period for calculating the mean and standard
                    deviation of returns.
    :return: (pd.Series or pd.DataFrame) -1, 0, or 1 denoting whether the return for each observation is
                    less/between/greater than the threshold at each corresponding time index. First or last row will be
                    NaN, depending on lag.
    """
    # Apply resample period, if applicable.
    if resample_by is not None:
        prices = prices.resample(resample_by).last()

    # Calculate returns.
    if lag:
        returns = prices.pct_change(1).shift(-1)
    else:
        returns = prices.pct_change(1)

    # If threshold is pd.Series, its index must patch prices.index; otherwise labels will fail to return.
    if isinstance(threshold, pd.Series):
        assert threshold.index.equals(prices.index), "prices.index and threshold.index must match! If prices are " \
                                                     "resampled, the threshold index must match the resampled prices " \
                                                     "index."

    # Adjust by mean and stdev, if desired. Assert that window must exist if standardization is on. Warning if window
    # is too large.
    if standardized:
        assert isinstance(window, int), "When standardized is True, window must be int."
        if window >= len(returns):
            warnings.warn('window is greater than the length of the Series. All labels will be NaN.', UserWarning)

        # Apply standardization.
        mean = returns.rolling(window=window).mean()
        stdev = returns.rolling(window=window).std()
        returns -= mean
        returns /= stdev

    # Apply labeling.
    labels = returns.copy()  # Copy returns so labels aren't all 0 when threshold => 1.
    labels[returns.lt(-threshold, axis=0)] = -1
    labels[returns.gt(threshold, axis=0)] = 1
    labels[(returns.ge(-threshold, axis=0)) & (returns.le(threshold, axis=0))] = 0

    return labels
