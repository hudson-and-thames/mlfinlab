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


def get_tick_rule(price: float, prev_price: float, prev_tick: int) -> int:
    """
    Tick rule calculation logic
    :param price: (float): current tick price
    :param prev_price: (float): previous tick price
    :param prev_tick: (int): previous tick value
    :return: (int): -1
    """
    if price > prev_price:
        tick_rule = 1
    elif price < prev_price:
        tick_rule = -1
    else:
        tick_rule = prev_tick
    return tick_rule
