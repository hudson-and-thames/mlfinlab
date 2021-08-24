"""
First generation features (Roll Measure/Impact, Corwin-Schultz spread estimator)
"""

import numpy as np
import pandas as pd


def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, page 282.

    Get Roll Measure

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) Estimation window
    :return: (pd.Series) Roll measure
    """

    pass


def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Impact.

    Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) Estimation window
    :return: (pd.Series) Roll impact
    """

    pass


# Corwin-Schultz algorithm
def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get beta estimate from Corwin-Schultz algorithm

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Beta estimates
    """

    pass


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get gamma estimate from Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) Gamma estimates
    """

    pass


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get alpha from Corwin-Schultz algorithm.

    :param beta: (pd.Series) Beta estimates
    :param gamma: (pd.Series) Gamma estimates
    :return: (pd.Series) Alphas
    """

    pass


def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get Corwin-Schultz spread estimator using high-low prices

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0

    pass


def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.2, page 286.

    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name

    pass
