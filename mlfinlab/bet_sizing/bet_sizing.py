"""
This module contains functionality for determining bet sizes for
investments based on machine learning predicetions. These implementations are
based on bet sizing approaches described in Chapter 10.
"""


# imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from mlfinlab.bet_sizing.ef3m import M2N
from mlfinlab.bet_sizing.ch10_snippets import (get_signal, avg_active_signals,
    discrete_signal)


def bet_size_probability(events, prob, pred, num_classes, step_size=0.0,
    average_active=False, num_threads=1):
    """
    Calculates the bet size using the predicted probability. Note that if
    'average_active' is True, the returned pandas.Series will be twice
    the length of the orginal since the average is calculated at each
    bet's open and close.

    :param prob: (pandas.Series) The predicted probabiility.
    :param pred: (pd.Series) The predicted bet side. 
    :param num_classes: (int) The number of predicted bet sides.
    :param step_size: (float) The step size at which the bet size is 
        discretized, default is 0.0 which imposes no discretization.
    :param average_active: (bool) Option to average the size of active bets.
    :param num_threads: (int) The number of processing threads to utilize for
        multiprocessing, default value is 1.
    :return: (pandas.Series) The bet size, with the time index.
    """
    signal_0 = get_signal(prob, pred, num_classes)
    df_0 = signal_0.to_frame('signal').join(events['t1'], how='left')

    if average_active:
        ser_0 = avg_active_signals(df_0, num_threads)
    else:
        ser_0 = df_0.signal
    if step_size > 0:
        ser_0 = discrete_signal(signal0=ser_0, step_size=step_size)
    
    return ser_0


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
