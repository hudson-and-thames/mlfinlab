"""
Return in excess of median method
Described in "The benefits of tree-based models for stock selection" Zhu et al. (2012). Data labelled this way
"""
import pandas as pd
import numpy as np


def excess_over_median(prices, binary=False):
    """
    Return in excess of median labelling method
    Returns a DataFrame containing returns of stocks over the median of all stocks in the portfolio, or a DataFrame of
    signs or excess return over median, if binary is True. In this case, an observation may be labelled as 0 if it
    itself is the median.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market that are used to establish the median.
                    Returns on each ticker are then compared to the median for the given timestamp.
    :param binary: (bool) If False, the numerical value of excess returns over median will be given. If True, then only
                    the sign of the excess return over median will be given (-1 or 1). A label of 0 will be given if
                    the observation itself is the median.
    :return: (pd.DataFrame) Numerical returns in excess of the market median return, or sign of return depending on
                whether binary is True or False. The last row will be NaN.
    """
    # Get return per period
    returns = prices.pct_change(periods=1).shift(-1)

    # Calculate median returns for each period as market return, if not specified
    market_return = returns.median(axis=1)

    # Calculate excess over market (median) return
    returns_over_median = returns.sub(market_return, axis=0)

    # If binary is true, returns sign of the return over median instead of the value
    if binary:
        return np.sign(returns_over_median)

    return returns_over_median
