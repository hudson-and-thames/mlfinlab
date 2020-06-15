"""
Labeling Raw Returns.

Most basic form of labeling based on raw return of each observation relative to its previous value.
"""

import numpy as np


def raw_return(prices, binary=False, logarithmic=False, resample_by=None, lag=True):
    """
    Raw returns labeling method.

    This is the most basic and ubiquitous labeling method used as a precursor to almost any kind of financial data
    analysis or machine learning. User can specify simple or logarithmic returns, numerical or binary labels, a
    resample period, and whether returns are lagged to be forward looking.

    :param prices: (pd.Series or pd.DataFrame) Time-indexed price data on stocks with which to calculate return.
    :param binary: (bool) If False, will return numerical returns. If True, will return the sign of the raw return.
    :param logarithmic: (bool) If False, will calculate simple returns. If True, will calculate logarithmic returns.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return:  (pd.Series or pd.DataFrame) Raw returns on market data. User can specify whether returns will be based on
                simple or logarithmic return, and whether the output will be numerical or categorical.
    """
    # Apply resample, if applicable.
    if resample_by is not None:
        prices = prices.resample(resample_by).last()

    # Get return per period.
    if logarithmic:  # Log returns
        if lag:
            returns = np.log(prices).diff().shift(-1)
        else:
            returns = np.log(prices).diff()
    else:  # Simple returns
        if lag:
            returns = prices.pct_change(periods=1).shift(-1)
        else:
            returns = prices.pct_change(periods=1)

    # Return sign only if categorical labels desired.
    if binary:
        returns = returns.apply(np.sign)

    return returns
