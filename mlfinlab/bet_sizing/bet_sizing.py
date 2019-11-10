"""
This module contains functionality for determining bet sizes for investments based on machine learning predictions.
These implementations are based on bet sizing approaches described in Chapter 10.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, moment

from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, discrete_signal
from mlfinlab.bet_sizing.ch10_snippets import get_w, get_target_pos, limit_price, bet_size
from mlfinlab.bet_sizing.ef3m import M2N, raw_moment, most_likely_parameters


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

    if abs(step_size) > 0:
        signal_1 = discrete_signal(signal0=signal_1, step_size=abs(step_size))

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
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
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


def bet_size_budget(events_t1, sides):
    """
    Calculates a bet size from the bet sides and start and end times. These sequences are used to determine the
    number of concurrent long and short bets, and the resulting strategy-independent bet sizes are the difference
    between the average long and short bets at any given time. This strategy is based on the section 10.2
    in "Advances in Financial Machine Learning". This creates a linear bet sizing scheme that is aligned to the
    expected number of concurrent bets in the dataset.

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with the number of concurrent
     active long and short bets, as well as the bet size, in additional columns.
    """
    events_1 = get_concurrent_sides(events_t1, sides)
    active_long_max, active_short_max = events_1['active_long'].max(), events_1['active_short'].max()
    frac_active_long = events_1['active_long'] / active_long_max if active_long_max > 0 else 0
    frac_active_short = events_1['active_short'] / active_short_max if active_short_max > 0 else 0
    events_1['bet_size'] = frac_active_long - frac_active_short

    return events_1


def bet_size_reserve(events_t1, sides, fit_runs=100, epsilon=1e-5, factor=5, variant=2, max_iter=10_000,
                     num_workers=1, return_parameters=False):
    """
    Calculates the bet size from bet sides and start and end times. These sequences are used to determine the number
    of concurrent long and short bets, and the difference between the two at each time step, c_t. A mixture of two
    Gaussian distributions is fit to the distribution of c_t, which is then used to determine the bet size. This
    strategy results in a sigmoid-shaped bet sizing response aligned to the expected number of concurrent long
    and short bets in the dataset.

    Note that this function creates a <mlfinlab.bet_sizing.ef3m.M2N> object and makes use of the parallel fitting
    functionality. As such, this function accepts and passes fitting parameters to the
    mlfinlab.bet_sizing.ef3m.M2N.mp_fit() method.

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :param fit_runs: (int) Number of runs to execute when trying to fit the distribution.
    :param epsilon: (float) Error tolerance.
    :param factor: (float) Lambda factor from equations.
    :param variant: (int) Which algorithm variant to use, 1 or 2.
    :param max_iter: (int) Maximum number of iterations after which to terminate loop.
    :param num_workers: (int) Number of CPU cores to use for multiprocessing execution, set to -1 to use all
     CPU cores. Default is 1.
    :param return_parameters: (bool) If True, function also returns a dictionary of the fited mixture parameters.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with the number of concurrent
     active long, short bets, the difference between long and short, and the bet size in additional columns.
     Also returns the mixture parameters if 'return_parameters' is set to True.
    """
    events_active = get_concurrent_sides(events_t1, sides)
    # Calculate the concurrent difference in active bets: c_t = <current active long> - <current active short>
    events_active['c_t'] = events_active['active_long'] - events_active['active_short']
    # Calculate the first 5 centered and raw moments from the c_t distribution.
    central_mmnts = [moment(events_active['c_t'].to_numpy(), moment=i) for i in range(1, 6)]
    raw_mmnts = raw_moment(central_moments=central_mmnts, dist_mean=events_active['c_t'].mean())
    # Fit the mixture of distributions.
    m2n = M2N(raw_mmnts, epsilon=epsilon, factor=factor, n_runs=fit_runs,
              variant=variant, max_iter=max_iter, num_workers=num_workers)
    df_fit_results = m2n.mp_fit()
    fit_params = most_likely_parameters(df_fit_results)
    params_list = [fit_params[key] for key in ['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'p_1']]
    # Calculate the bet size.
    events_active['bet_size'] = events_active['c_t'].apply(lambda c: single_bet_size_mixed(c, params_list))

    if return_parameters:
        return events_active, fit_params
    return events_active


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
                d_vars[k] = pd.Series(data=np.array([d_vars[k] for i in range(ser_len)]), index=idx)

    # Combine Series to form a DataFrame.
    events = pd.concat(list(d_vars.values()), axis=1)
    events.columns = list(d_vars.keys())

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


def cdf_mixture(x_val, parameters):
    """
    The cumulative distribution function of a mixture of 2 normal distributions, evaluated at x_val.

    :param x_val: (float) Value at which to evaluate the CDF.
    :param parameters: (list) The parameters of the mixture, [mu_1, mu_2, sigma_1, sigma_2, p_1]
    :return: (float) CDF of the mixture.
    """
    mu_1, mu_2, sigma_1, sigma_2, p_1 = parameters  # Parameters reassigned for clarity.
    return p_1 * norm.cdf(x_val, mu_1, sigma_1) + (1-p_1) * norm.cdf(x_val, mu_2, sigma_2)


def single_bet_size_mixed(c_t, parameters):
    """
    Returns the single bet size based on the description provided in question 10.4(c), provided the difference in
    concurrent long and short positions, c_t, and the fitted parameters of the mixture of two Gaussain distributions.

    :param c_t: (int) The difference in the number of concurrent long bets minus short bets.
    :param parameters: (list) The parameters of the mixture, [mu_1, mu_2, sigma_1, sigma_2, p_1]
    :return: (float) Bet size.
    """
    if c_t >= 0:
        single_bet_size = (cdf_mixture(c_t, parameters) - cdf_mixture(0, parameters)) / (1 - cdf_mixture(0, parameters))
    else:
        single_bet_size = (cdf_mixture(c_t, parameters) - cdf_mixture(0, parameters)) / cdf_mixture(0, parameters)
    return single_bet_size
