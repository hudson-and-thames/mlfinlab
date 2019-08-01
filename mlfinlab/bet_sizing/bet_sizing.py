"""
This module contains functionality for determining bet sizes for investments based on machine learning predictions.
These implementations are based on bet sizing approaches described in Chapter 10.
"""

import numpy as np
import pandas as pd

from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, discrete_signal
from mlfinlab.bet_sizing.ch10_snippets import get_w, get_target_pos, limit_price, bet_size


def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0, average_active=False, num_threads=1):
    """
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.

    :param events: (pandas.DataFrame) Contains at least the column 't1', the expiry datetime of the product, with
     a datetime index, the datetime the position was taken.
    :param prob: (pandas.Series) The predicted probability.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :param step_size: (float) The step size at which the bet size is discretized, default is 0.0 which imposes no
     discretization.
    :param average_active: (bool) Option to average the size of active bets, default value is False.
    :param num_threads: (int) The number of processing threads to utilize for multiprocessing, default value is 1.
    :return: (pandas.Series) The bet size, with the time index.
    """
    signal_0 = get_signal(prob, num_classes, pred)
    events_0 = signal_0.to_frame('signal').join(events['t1'], how='left')
    if average_active:
        signal_1 = avg_active_signals(events_0, num_threads)
    else:
        signal_1 = events_0.signal

    if step_size > 0:
        signal_1 = discrete_signal(signal0=signal_1, step_size=step_size)

    return signal_1


def bet_size_dynamic(current_pos, max_pos, market_price, forecast_price, cal_divergence=10, cal_bet_size=0.95,
                     func='sigmoid'):
    """
    Calculates the bet sizes, target position, and limit price as the market price and forecast price fluctuate.
    The current position, maximum position, market price, and forecast price can be passed as separate pandas.Series
    (with a common index), as individual numbers, or a combination thereof. If any one of the aforementioned arguments
    is a pandas.Series, the other arguments will be broadcast to a pandas.Series of the same length and index.

    :param current_pos: (pandas.Series, int) Current position.
    :param max_pos: (pandas.Series, int) Maximum position
    :param market_price: (pandas.Series, float) Market price.
    :param forecast_price: (pandas.Series, float) Forecast price.
    :param cal_divergence: (float) The divergence to use in calibration.
    :param cal_bet_size: (float) The bet size to use in calibration.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid'.
    :return: (pandas.DataFrame) Bet size (bet_size), target position (t_pos), and limit price (l_p).
    """
    # Create a dictionary of bet size variables for easier handling.
    d_vars = {'pos': current_pos, 'max_pos': max_pos, 'm_p': market_price, 'f': forecast_price}
    events_0 = confirm_and_cast_to_df(d_vars)

    # Calibrate w.
    w_param = get_w(cal_divergence, cal_bet_size, func)
    # Compute the target bet position.
    events_0['t_pos'] = events_0.apply(lambda x: get_target_pos(w_param, x.f, x.m_p, x.max_pos, func), axis=1)
    # Compute the break even limit price.
    events_0['l_p'] = events_0.apply(lambda x: limit_price(x.t_pos, x.pos, x.f, w_param, x.max_pos, func), axis=1)
    # Compute the bet size.
    events_0['bet_size'] = events_0.apply(lambda x: bet_size(w_param, x.f-x.m_p, func), axis=1)

    return events_0[['bet_size', 't_pos', 'l_p']]


def confirm_and_cast_to_df(d_vars):
    """
    Accepts either pandas.Series (with a common index) or integer/float values, casts all non-pandas.Series values
    to Series, and returns a pandas.DataFrame for further calculations. This is a helper function to the
    'bet_size_dynamic' function.

    :param d_vars: (dict) A dictionary where the values are either pandas.Series or single int/float values.
     All pandas.Series passed are assumed to have the same index. The keys of the dictionary will be used for column
     names in the returned pandas.DataFrame.
    :return: (pandas.DataFrame) The values from the input dictionary in pandas.DataFrame format, with dictionary
     keys as column names.
    """
    any_series = False  # Are any variables a pandas.Series?
    all_series = True  # Are all variables a pandas.Series?
    ser_len = 0
    for var in d_vars.values():
        any_series = any_series or isinstance(var, pd.Series)
        all_series = all_series and isinstance(var, pd.Series)

        if isinstance(var, pd.Series):
            ser_len = var.size
            idx = var.index

    # Handle data types if there are no pandas.Series variables.
    if not any_series:
        for k in d_vars:
            d_vars[k] = pd.Series(data=[d_vars[k]], index=[0])

    # Handle data types if some but not all variables are pandas.Series.
    if any_series and not all_series:
        for k in d_vars:
            if not isinstance(d_vars[k], pd.Series):
                print(f"{k} becomes a pd.Series from a {type(d_vars[k])}")
                d_vars[k] = pd.Series(data=np.array([d_vars[k] for i in range(ser_len)]), index=idx)

    # Combine Series to form a DataFrame.
    events = pd.concat([d_vars.values()], axis=1)
    d_col_names = {i: k_i for i, k_i in enumerate(d_vars.keys())}
    events = events.rename(columns=d_col_names)

    return events

def get_concurrent_sides(events_t1, sides):
    """
    Given the side of the position along with its start and end timestamps, this function returns two pandas.Series
    indicating the number of concurrent long and short bets at each timestamp.

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with two additional columns
     indicating the number of concurrent active long and active short bets at each timestamp.
    """
    events_0 = pd.DataFrame({'t1':events_t1, 'side':sides})
    events_0['active_long'] = 0
    events_0['active_short'] = 0

    for idx in events_0.index:
        # A bet side greater than zero indicates a long position.
        df_long_active_idx = set(events_0[(events_0.index <= idx) & (events_0['t1'] > idx) & (events_0['side'] > 0)].index)
        events_0.loc[idx, 'active_long'] = len(df_long_active_idx)
        # A bet side less than zero indicates a short position.
        df_short_active_idx = set(events_0[(events_0.index <= idx) & (events_0['t1'] > idx) & (events_0['side'] < 0)].index)
        events_0.loc[idx, 'active_short'] = len(df_short_active_idx)

    return events_0

def bet_size_budget(events_t1, sides):
    """
    Calculates a bet size from the bet sides and start and end times. These sequences are used to determine the
    number of concurrent long and short bets, and the resulting strategy-independent bet sizes are the difference
    between the average long and short bets at any given time. This strategy is based on the section 10.2
    in "Advances in Financial Machine Learning".

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with the number of concurrent
     active long and short bets, as well as the bet size, in additional columns.
    """
    events_1 = get_concurrent_sides(events_t1, sides)
    avg_active_long = events_1['active_long'] / events_1['active_long'].max()
    avg_active_short = events_1['active_short'] / events_1['active_short'].max()
    events_1['bet_size'] = avg_active_long - avg_active_short

    return events_1
