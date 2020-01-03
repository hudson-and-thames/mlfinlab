"""
Volume classification methods (BVC and tick rule)
"""

from scipy.stats import norm
import pandas as pd


def get_bvc_buy_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """

    :param close: (pd.Series): series of close prices
    :param volume: (pd.Series): series of bar volumes
    :param window: (int); window for std estimation uses in BVC calculation
    :return:
    """
    return volume * norm.cdf(close.diff() / close.diff().rolling(window=window).std())
