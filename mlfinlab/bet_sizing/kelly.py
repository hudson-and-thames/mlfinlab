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
    Calculates optimal bet size based on the kelly criterion.

    This method assumes our strategy is used for the environment which have discrete and binary outcomes for each trial
    such as coin tossing and sports betting. It returns the optimal betting size based on the kelly criterion.

    This method is designed to deal with not only the scenario that we loss all of wager when we loss(loss_unit=1)
    but also the general scenario. For example, when betting odds are 2, we should set 'profit_unit' to 2 and
    'loss_unit' to 1.

    The implementation is based on the paper, 'The Kelly Criterion in blackjack, sports betting, and the stock market'
    by Edward O. Thorp.

    :param win_probability: (float) This is probability of a win of your strategy.
    :param profit: (float) profit per unit bet.
    :param loss: (float) loss per unit bet.
    :return: (kelly_bet_size, expected_growth_rate) (tuple[float, float])
            kelly_bet_size: (float) optimal bet size from kelly criterion.
            expected_growth_rate: (float) (expected) growth rate of balance(capital) when you bet as much as kelly bet
            size.
    """
    loss_probability = 1 - win_probability

    kelly_bet_size = win_probability / loss_unit - loss_probability / profit_unit
    expected_growth_rate = loss_probability * np.log(1 - loss_unit * kelly_bet_size) + \
                           win_probability * np.log(1 + profit_unit * kelly_bet_size)

    return kelly_bet_size, expected_growth_rate


def kelly_investing(returns: np.ndarray, risk_free_rate: float, annualize_factor: int) -> Tuple[float, float]:
    """
    Calculates optimal leverage of investing on a single asset based on the kelly criterion.

    This method assumes our strategy is used for a single asset in the environment where there is not a finite number
    of outcomes like investing on securities in the stock market.

    The implementation is based on the paper, 'The Kelly Criterion in blackjack, sports betting, and the stock market'
    by Edward O. Thorp.

    :param returns: (np.ndarray) This is series of returns of your strategy or a asset.
    :param risk_free_rate: (float) this is the annualized return on the remaining capital which is not invested on your
    main strategy. The rate of the risk free asset such as treasury bill rate is usually used.
    :param annualize_factor: (int) It is a scale factor to estimate annualized return of the strategy which depends on
    the period of returns.

    :return: (kelly_leverage, expected_growth_rate) (tuple[float, float])
            kelly_leverage: (float) optimal leverage size from kelly criterion.
            expected_growth_rate: (float) (expected) growth rate of balance(capital) when you invest as much as optimal
            leverage calculated from kelly criterion.
    """
    mean = np.mean(returns) * annualize_factor
    var = np.var(returns, ddof=1) * annualize_factor

    excess_mean = mean - risk_free_rate

    kelly_leverage = excess_mean / var
    expected_growth_rate = risk_free_rate + (excess_mean ** 2) / (2 * var)

    return kelly_leverage, expected_growth_rate


def kelly_allocation(returns: pd.DataFrame, risk_free_rate: float, annualize_factor: int) -> Tuple[np.ndarray, float]:
    """
    Calculates optimal leverage among the multiple assets or strategies based on the kelly criterion.

    The implementation is based on the paper, 'The Kelly Criterion in blackjack, sports betting, and the stock market'
    by Edward O. Thorp.

    :param returns: (pd.DataFrame) 2-d array of returns. The columns are assets(strategies) and rows are time.
    :param risk_free_rate: (float) this is the annualized return on the remaining capital which is not invested on your
    main strategy. The rate of the risk free asset such as treasury bill rate is usually used.
    :param annualize_factor: (int) It is a scale factor to estimate annualized return of the strategy which depends on
    the period of returns.

    :return: (fractions, expected_growth_rate) (tuple[float, float])
            fractions: (np.ndarray) An array which contains optimal fraction(leverage) of each asset(strategy)
            from kelly criterion.
            expected_growth_rate: (float) (expected) growth rate of balance(capital) when you distribute your wealth
            according to kelly criterion.
    """
    means = returns.mean()*annualize_factor
    cov_matrix = returns.cov()*annualize_factor

    fractions = np.dot(inv(cov_matrix), means-risk_free_rate)
    expected_growth_rate = risk_free_rate + np.dot(fractions.T, cov_matrix.dot(fractions))/2

    return fractions, expected_growth_rate
