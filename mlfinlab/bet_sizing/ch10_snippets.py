"""
This module contains the code snippets found in Chapter 10 of "Advances in Financial Machine Learning" by
Marcos López de Prado. The code has been amended for readability, to conform to PEP8 rules, to keep the snippets as
manageable single-units of functionality, as well as to account for deprecation of functions originally used, but is
otherwise unaltered.
"""

import warnings
import pandas as pd
import numpy as np
from scipy.stats import norm

from mlfinlab.util.multiprocess import mp_pandas_obj


def get_signal(prob, num_classes, pred=None):
    """
    SNIPPET 10.1 - FROM PROBABILITIES TO BET SIZE
    Calculates the given size of the bet given the side and the probability (i.e. confidence) of the prediction. In this
    representation, the probability will always be between 1/num_classes and 1.0.

    :param prob: (pd.Series) The probability of the predicted bet side.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :return: (pd.Series) The bet size.
    """

    pass


def avg_active_signals(signals, num_threads=1):
    """
    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
    Function averages the bet sizes of all concurrently active bets. This function makes use of multiprocessing.

    :param signals: (pandas.DataFrame) Contains at least the following columns:
     'signal' - the bet size
     't1' - the closing time of the bet
     And the index must be datetime format.
    :param num_threads: (int) Number of threads to use in multiprocessing, default value is 1.
    :return: (pandas.Series) The averaged bet sizes.
    """

    pass


def mp_avg_active_signals(signals, molecule):
    """
    Part of SNIPPET 10.2
    A function to be passed to the 'mp_pandas_obj' function to allow the bet sizes to be averaged using multiprocessing.

    At time loc, average signal among those still active.
    Signal is active if (a) it is issued before or at loc, and (b) loc is before the signal's end time,
    or end time is still unknown (NaT).

    :param signals: (pandas.DataFrame) Contains at least the following columns: 'signal' (the bet size) and 't1' (the closing time of the bet).
    :param molecule: (list) Indivisible tasks to be passed to 'mp_pandas_obj', in this case a list of datetimes.
    :return: (pandas.Series) The averaged bet size sub-series.
    """

    pass


def discrete_signal(signal0, step_size):
    """
    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING
    Discretizes the bet size signal based on the step size given.

    :param signal0: (pandas.Series) The signal to discretize.
    :param step_size: (float) Step size.
    :return: (pandas.Series) The discretized signal.
    """

    pass


# ==============================================================================
# SNIPPET 10.4 - DYNAMIC POSITION SIZE AND LIMIT PRICE
# The below functions are part of or derived from the functions
# in snippet 10.4.
# ==============================================================================
# Bet size calculations based on a sigmoid function.
def bet_size_sigmoid(w_param, price_div):
    """
    Part of SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a sigmoid function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, forecast price - market price.
    :return: (float) The bet size.
    """

    pass


def get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a sigmoid function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :return: (int) Target position.
    """

    pass


def inv_price_sigmoid(forecast_price, w_param, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a sigmoid function for a bet size algorithm.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """

    pass


def limit_price_sigmoid(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the limit price.
    Based on a sigmoid function for a bet size algorithm.

    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :return: (float) Limit price.
    """

    pass


def get_w_sigmoid(price_div, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    Based on a sigmoid function for a bet size algorithm.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """

    pass


# ==============================================================================
# Bet size calculations based on a power function.
def bet_size_power(w_param, price_div):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a power function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price, must be between -1 and 1, inclusive.
    :return: (float) The bet size.
    """

    pass


def get_target_pos_power(w_param, forecast_price, market_price, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a power function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Target position.
    """

    pass


def inv_price_power(forecast_price, w_param, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a power function for a bet size algorithm.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """

    pass


def limit_price_power(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. Based on a power function for a bet size algorithm.

    :param target_pos: (float) Target position.
    :param pos: (float) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Limit price.
    """

    pass


def get_w_power(price_div, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'w' coefficient must be greater than or equal to zero.
    Based on a power function for a bet size algorithm.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """

    pass


# ==============================================================================
# Bet size calculation functions, power and sigmoid packaged together.
# This is useful as more bet sizing function options are added.
def bet_size(w_param, price_div, func):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) The bet size.
    """

    pass


def get_target_pos(w_param, forecast_price, market_price, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. The 'func' argument allows the user to choose between bet sizing functions.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (int) Target position.
    """

    pass


def inv_price(forecast_price, w_param, m_bet_size, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """

    pass


def limit_price(target_pos, pos, forecast_price, w_param, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. The 'func' argument allows the user to choose between bet sizing functions.

    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) Limit price.
    """

    pass


def get_w(price_div, m_bet_size, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """

    pass
