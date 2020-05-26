"""
An implementation of applications of Kelly Criterion. Based on the work by Edward O. Thorp. (2006)
"The Kelly Criterion in blackjack, sports betting, and the stock market" Handbook of Asset and Liability Management,
Vol. 1, 387-428.
"""

from typing import Tuple

import numpy as np
from numpy.linalg import inv
import pandas as pd


def kelly_betting(win_probability: float, profit_unit: float, loss_unit: float) -> Tuple[float, float]:
    """
    Calculates optimal bet size based on the Kelly criterion.

    This method assumes our strategy is used for the environment which has discrete and binary outcomes for each trial
    such as coin tossing and sports betting. It returns the optimal betting size based on the Kelly criterion.

    This method is designed to deal with not only the scenario that we lose all of the wager when we lose(loss_unit=1)
    but also the general scenario. For example, when betting odds are 2, we should set 'profit_unit' to 2 and
    'loss_unit' to 1.

    The implementation is based on the paper, 'The Kelly Criterion in blackjack, sports betting, and the stock market'
    by Edward O. Thorp.

    :param win_probability: (float) Probability of win for a strategy.
    :param profit_unit: (float) Profit per unit bet.
    :param loss_unit: (float) Loss per unit bet.
    :return: (float, float) The first element is the optimal bet size from the Kelly criterion. The second is the
    expected compound growth rate of balance(capital) when the bet is the Kelly bet size.
    """
    loss_probability = 1 - win_probability

    kelly_bet_size = win_probability / loss_unit - loss_probability / profit_unit
    expected_growth_rate = loss_probability * np.log(1 - loss_unit * kelly_bet_size) + \
                           win_probability * np.log(1 + profit_unit * kelly_bet_size)

    return kelly_bet_size, expected_growth_rate


def kelly_investing(returns: np.ndarray, risk_free_rate: float, annualize_factor: int, raw_return: bool = True) \
        -> Tuple[float, float]:
    """
    Calculates optimal leverage of investing in a single asset based on the Kelly criterion.

    This method assumes our strategy is used for a single asset in the environment where there is not a finite number
    of outcomes like investing in securities in the stock market.

    The implementation is based on the paper, 'The Kelly Criterion in blackjack, sports betting, and the stock market'
    by Edward O. Thorp.

    :param returns: (np.ndarray) Returns of your strategy or a asset.
    :param risk_free_rate: (float) Annual return of the risk-free asset.
    :param annualize_factor: (int) Scale factor to estimate annualized return of the strategy or asset.
    :param raw_return: (bool) Flag whether returns are raw return or log return.
    :return: (float, float) The first element is the optimal leverage from the Kelly criterion. The second is the
    expected compound growth rate of balance(capital) from the Kelly criterion.
    """
    if raw_return:
        returns = np.log(1+returns)
    mean = np.mean(returns) * annualize_factor
    var = np.var(returns, ddof=1) * annualize_factor

    excess_mean = mean - risk_free_rate

    kelly_leverage = excess_mean / var
    expected_growth_rate = risk_free_rate + (excess_mean ** 2) / (2 * var)

    return kelly_leverage, expected_growth_rate


def kelly_allocation(returns: pd.DataFrame, risk_free_rate: float, annualize_factor: int, raw_return: bool = True) \
        -> Tuple[np.ndarray, float]:
    """
    Calculates optimal leverage among the multiple assets or strategies based on the Kelly criterion.

    The implementation is based on the paper, 'The Kelly Criterion in blackjack, sports betting, and the stock market'
    by Edward O. Thorp.

    :param returns: (pd.DataFrame) Dataframe where each column is a series of returns for an asset(strategy).
    :param risk_free_rate: (float) Annual return of the risk-free asset.
    :param annualize_factor: (int) Scale factor to estimate annualized return of the strategy or asset.
    :param raw_return: (bool) Flag whether returns are raw return or log return.
    :return: (np.ndarray, float) The first element is fractions(leverage) of each asset(strategy) from the Kelly
    criterion. The second is the expected compound growth rate of balance(capital) by the Kelly criterion.
    """
    if raw_return:
        returns = np.log(1+returns)

    means = returns.mean() * annualize_factor
    cov_matrix = returns.cov() * annualize_factor

    fractions = np.dot(inv(cov_matrix), means-risk_free_rate)
    expected_growth_rate = risk_free_rate + np.dot(fractions.T, cov_matrix.dot(fractions))/2

    return fractions, expected_growth_rate
