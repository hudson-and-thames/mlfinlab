"""
Second generation models features: Kyle lambda, Amihud Lambda, Hasbrouck lambda (bar and trade based)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# pylint: disable=invalid-name
def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Kyle lambda from bars data, p.286-288.

    :param close: (pd.Series) Close prices
    :param volume: (pd.Series) Bar volume
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Kyle lambdas
    """
    close_diff = close.diff()
    close_diff_sign = np.sign(close_diff)
    close_diff_sign.replace(0, method='pad', inplace=True)  # Replace 0 values with previous
    volume_mult_trade_signs = volume * close_diff_sign  # bt * Vt
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()


def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Amihud lambda from bars data, p.288-289.

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Amihud lambda
    """
    returns_abs = np.log(close / close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()


def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Hasbrouck lambda from bars data, p.289-290.

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Hasbrouck lambda
    """
    log_ret = np.log(close / close.shift(1))
    log_ret_sign = np.sign(log_ret).replace(0, method='pad')

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()


def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> float:
    """
    Get Kyle lambda from trades data, p.286-288.

    :param price_diff: (list) of price diffs
    :param volume: (list) of trades sizes
    :param aggressor_flags: (list) of trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (float) Kyle lambda for a bar
    """
    model = LinearRegression(fit_intercept=False, copy_X=False)
    signed_volume = np.array(volume) * np.array(aggressor_flags)
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    model.fit(X, y)
    return model.coef_[0]


def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> float:
    """
    Get Amihud lambda from trades data, p.288-289.

    :param log_ret: (list) of log returns
    :param dollar_volume: (list) of dollar volumes (price * size)
    :return: (float) Amihud lambda for a bar
    """
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]


def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> float:
    """
    Get Amihud lambda from trades data, p.289-290.

    :param log_ret: (list) of log returns
    :param dollar_volume: (list) of dollar volumes (price * size)
    :param aggressor_flags: (list) of trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (float) Amihud lambda for a bar
    """
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]
