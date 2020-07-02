"""
Return in excess of mean method.

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).
"""
import numpy as np


def excess_over_mean(prices, binary=False, resample_by=None, lag=True):
    """
    Return in excess of mean labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,
    by Coqueret, G. and Guida, T. (2020).

    Returns a DataFrame containing returns of stocks over the mean of all stocks in the portfolio. Returns a DataFrame
    of signs of the returns if binary is True. In this case, an observation may be labeled as 0 if it itself is the
    mean.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market that are used to establish the mean. NaN
                    values are ok. Returns on each ticker are then compared to the mean for the given timestamp.
    :param binary: (bool) If False, the numerical value of excess returns over mean will be given. If True, then only
                    the sign of the excess return over mean will be given (-1 or 1). A label of 0 will be given if
                    the observation itself equal to the mean.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.DataFrame) Numerical returns in excess of the market mean return, or sign of return depending on
                whether binary is False or True respectively.
    """
    # Apply resample, if applicable.
    if resample_by is not None:
        prices = prices.resample(resample_by).last()

    # Get return per period.
    if lag:
        returns = prices.pct_change(periods=1).shift(-1)
    else:
        returns = prices.pct_change(periods=1)

    # Calculate median returns for each period as market return.
    market_return = returns.mean(axis=1)

    # Calculate excess over market (median) return.
    returns_over_mean = returns.sub(market_return, axis=0)

    # If binary is true, returns sign of the return over median instead of the value.
    if binary:
        returns_over_mean = returns_over_mean.apply(np.sign)

    return returns_over_mean
