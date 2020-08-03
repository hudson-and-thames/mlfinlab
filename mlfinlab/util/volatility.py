"""
Various volatility estimators
"""
import pandas as pd
import numpy as np


# pylint: disable=redefined-builtin

def get_daily_vol(close, lookback=100):
    """
    Advances in Financial Machine Learning, Snippet 3.1, page 44.

    Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    See the pandas documentation for details on the pandas.Series.ewm function.
    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: (pd.Series) Closing prices
    :param lookback: (int) Lookback period to compute volatility
    :return: (pd.Series) Daily volatility value
    """

    pass


def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator

    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Parkinson volatility
    """

    pass


def get_garman_class_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20) -> pd.Series:
    """
    Garman-Class volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Garman-Class volatility
    """

    pass


def get_yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20) -> pd.Series:
    """

    Yang-Zhang volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """

    pass
