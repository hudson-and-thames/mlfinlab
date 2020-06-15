"""
Return in excess of median method.

Described in "The benefits of tree-based models for stock selection", Zhu et al. (2012). Data labeled this way can be
used in regression and classification models to predict stock returns over market.
"""
import numpy as np


def excess_over_median(prices, binary=False, resample_by=None, lag=True):
    """
    Return in excess of median labeling method. Sourced from "The benefits of tree-based models for stock selection"
    Zhu et al. (2012).

    Returns a DataFrame containing returns of stocks over the median of all stocks in the portfolio, or returns a
    DataFrame containing signs of those returns. In the latter case, an observation may be labeled as 0 if it itself is
    the median.

    :param prices: (pd.DataFrame) Close prices of all stocks in the market that are used to establish the median.
                   Returns on each stock are then compared to the median for the given timestamp.
    :param binary: (bool) If False, the numerical value of excess returns over median will be given. If True, then only
                    the sign of the excess return over median will be given (-1 or 1). A label of 0 will be given if
                    the observation itself is the median. According to Zhu et al., categorical labels can alleviate
                    issues with extreme outliers present with numerical labels.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.DataFrame) Numerical returns in excess of the market median return, or sign of return depending on
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
    market_return = returns.median(axis=1)

    # Calculate excess over market (median) return.
    returns_over_median = returns.sub(market_return, axis=0)

    # If binary is true, returns sign of the return over median instead of the value.
    if binary:
        returns_over_median = returns_over_median.apply(np.sign)

    return returns_over_median
