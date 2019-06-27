"""
This module contains functionality for determining bet sizes for
investments based on machine learning predicetions. These implementations are
based on bet sizing approaches described in Chapter 10.
"""


# imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from ef3m import M2N


def get_bet_size(data, ts=None, t_end=None, p=None, algorithm=None,
    mixed_gaussian=False, discretize=0.0, average_active=False,
    func='sigmoid', forecast_price=None, market_price=None):
    """
    Returns a DataFrame with an added column for bet size.

    :param data: (pandas.DataFrame) containing data for bet sizing
    :param ts: (string) column name of timestamp, 'index' uses the
    index of the DataFrame
    :param p: (string) column name of bet side probability from prediction
    :param algorithm: (string) algorithm to use for finding bet size, valid
    options are:
        'probability' - determine bet size based on bet side probability (p. 142)
        'dynamic' - dynamic calculation of bet size (p. 145)
        'budget' - based on the number of concurrent long and short bets (p. 142)
    :param mixed_gaussian: (bool) when using algorithm 'budget', setting this
    option will use a mixture of two Gaussian distribution to calculate the
    bet size (p. 142)
    :param discretize: (float) when using algorithm 'probability', this option
    will determine the step size, between 0 and 1, to discretize the bet
    size. Default value is 0.0 which will result in a continuous 
    bet size (no discretization)
    :param average_active: (bool) when using algorithm 'probability', setting
    this option will average the size of all active bets
    :param func: (string) function to use for dynamic calculation. Valid
    options are:
        'sigmoid'
        'power'
    :return: (pandas.DataFrame) with the added bet size column
    """
    if algorithm == 'probability':
        # check for necessary options to execute 'probability' algorithm
        # return bet_size calculated using predicted probability
        data = bet_size_probability(data, p, discretize, average_active)
    
    elif algorithm == 'dynamic':
        # check for necessary options to execute 'dynamic' algorithm
        # return dynamic bet sizes
        data = bet_size_dynamic(data, market_price, forecast_price, func)
    
    elif algorithm == 'budget':
        # check for necessary options to execute 'budget' algorithm
        # return bet sizes based on budget calculations
        data = bet_size_budget(data, ts, t_end, mixed_gaussian)
    
    else:
        # handle case where there is no algorithm
        raise ValueError("Argument 'algorithm' must have a valid value, "
                          "can be one of: 'probability', 'dynamic', 'budget'")

    return data


def bet_size_probability(data, p, discretize=0.0, average_active=False):
    """
    Calculates the bet size using the predicted probability.

    :param data: (pandas.DataFrame) containing data for bet sizing
    :param p: (string) the column name of the predicted probability
    :param discretize: (float) the step size at which the bet size is 
    discretized, default is 0.0 which imposes no discretization
    :param average_active: (bool) option to average the size of active bets
    :return: (pandas.DataFrame) with the added bet size column
    """
    return data


def bet_size_dynamic(data, market_price, forecast_price, func='sigmoid'):
    """
    Calculates the bet sizes as the market price and forecast price fluctuate.

    :param data: (pandas.DataFrame) containing data for bet sizing
    :param market_price: (string) column name of the market price
    :param forecast_price: (string) column name of the forecast price
    :param func: (string) function to use for dynamic calculation. Valid
    options are:
        'sigmoid'
        'power'
    :return: (pandas.DataFrame) with the added bet size column
    """
    return data

def bet_size_budget(data, ts, t_end, mixed_gaussian=False):
    """
    Calculates the bet sizes based on a budgeting approach. Optionally
    fits a mixture of 2 Gaussians to the distribution of the number of
    concurrent long and short bets and uses the CDF of the mixture to
    calculate bet size.

    :param data: (pandas.DataFrame) containing data for bet sizing
    :param ts: (string) column name of the timestamp, default 'index'
    uses the index of the DataFrame
    :param mixed_gaussian: (bool) when using algorithm 'budget', setting this
    option will use a mixture of two Gaussian distribution to calculate the
    bet size (p. 142)
    :return: (pandas.DataFrame) with the added bet size column
    """
    return data
