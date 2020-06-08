"""
Labeling according to forward returns.

Work "Financial time series forecasting using support vector machines" by Kim, K. (2002) describes how
labeling data this way can be used to train a support vector machine to predict price movements.
"""
import warnings
import numpy as np
import pandas as pd


def forward_return(prices, lookforward=1):
    """
    Future returns labeling method.

    Returns 1 if next day's (or days') price is greater than current day's, and 0 otherwise.

    :param prices: (pd.Series or pd.DataFrame) Close prices.
    :param lookforward: (int) Number of ticks forward to compare current return to. (1 by default)
    :return: (pd.Series or pd.DataFrame) Labels of 1 if the forward price exceeds the current price, and 0 otherwise.
    """
    # Error if lookforward isn't an int, and warns if it exceeds the number of rows in prices
    assert isinstance(lookforward, int), "lookforward period must be int!"
    if lookforward >= len(prices):
        warnings.warn('lookforward is greater than number of rows in prices. All labels will be NaN.', UserWarning)

    # Get returns
    returns = prices.pct_change(lookforward).shift(-lookforward)

    # Get sign of returns
    returns = returns.apply(np.sign)

    # Apply labeling conditions
    conditions = [returns > 0, returns <= 0]
    choices = [1, 0]
    labels = np.select(conditions, choices, default=np.nan)

    # Returns as a pd.Series or pd.DataFrame
    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame(labels, index=prices.index, columns=prices.columns)
    return pd.Series(labels, index=prices.index)
