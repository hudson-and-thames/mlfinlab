"""
First generation features (Roll Measure/Impact, Corwin-Schultz spread estimator)
"""

import numpy as np
import pandas as pd


def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Measure (p.282, Roll Model). Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) estimation window
    :return: (pd.Series) of Roll measure
    """
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))


def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Impact. Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) estimation window
    :return: (pd.Series) of Roll impact
    """
    roll_measure = get_roll_measure(close_prices, window)
    return roll_measure / dollar_volume


# Corwin-Schultz algorithm
def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Get beta estimate from Corwin-Schultz algorithm (p.285, Snippet 19.1).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) estimation window
    :return: (pd.Series) of beta estimates
    """
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Get gamma estimate from Corwin-Schultz algorithm (p.285, Snippet 19.1).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) of gamma estimates
    """
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Get alpha from Corwin-Schultz algorithm, (p.285, Snippet 19.1).

    :param beta: (pd.Series) of beta estimates
    :param gamma: (pd.Series) of gamma estimates
    :return: (pd.Series) of alphas
    """
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha


def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Corwin-Schultz spread estimator using high-low prices, (p.285, Snippet 19.1).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) estimation window
    :return: (pd.Series) of Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread


def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm, (p.286, Snippet 19.2).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) estimation window
    :return: (pd.Series) of Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma
