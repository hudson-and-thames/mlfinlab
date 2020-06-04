"""
Most basic form of labeling based on raw return of each observation relative to its previous value. User can specify
percentage or logarithmic, numerical or categorical labels, and number of steps back to calculate the raw return from.
"""

import warnings
import numpy as np


def raw_return(price, binary=False, logarithmic=False, lookback=1):
    """
    Raw returns labeling method.

    This is the most basic and ubiquitous labeling method used as a precursor to almost any kind of financial data
    analysis or machine learning. User can specify percentage/logarithmic returns, numerical/binary labels, and
    lookback period to calculate the return from.

    :param price: (pd.Series or pd.DataFrame) Price data for one (Series) or multiple tickers (DataFrame).
    :param binary: (bool) If False, will return numerical returns. If True, will return the sign of the raw return.
                    (False by default).
    :param logarithmic: (bool) If False, will calculate percentage returns. If True, will calculate logarithmic returns.
                        (False by default).
    :param lookback: (int) Lookback period to calculate each observation's return from. 1 by default. The first
                    lookback number of rows in the output will be NaN.
    :return:  (pd.Series or pd.DataFrame) Raw returns on market data. User can specify whether returns will be based on
                percentage or logarithmic return, and whether the output will be numerical or categorical.
    """
    # Warning if lookback is greater than the number of rows
    if lookback >= len(price):
        warnings.warn('lookback period is greater than the length of the Series. All labels will be NaN.',
                      UserWarning)

    # Log returns or percentage returns
    if logarithmic:
        raw_returns = np.log(price) - np.log(price.shift(lookback))
    if not logarithmic:
        raw_returns = price.pct_change(lookback)

    # Return sign only if categorical labels desired
    if binary:
        raw_returns = np.sign(raw_returns)

    return raw_returns
