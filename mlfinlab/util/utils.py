"""
General functions that are used throughout research
"""


def get_daily_returns(intraday_returns):
    """
    Pyfolio cant create a performance tearsheet if you pass the function intraday returns.
    Thus we need to downsample to daily returns and pass that to pyfolio.

    This function takes intraday returns and returns the daily simple returns

    :param intraday_returns: Series of intraday returns
    :return: series of daily simple returns
    """
    # Cumulate the returns to create an index
    cum_rets = ((intraday_returns + 1).cumprod())

    # Downsample to daily
    daily_rets = cum_rets.resample('B').last()

    # Forward fill, Percent Change, Drop NaN
    daily_rets = daily_rets.ffill().pct_change().dropna()

    return daily_rets
