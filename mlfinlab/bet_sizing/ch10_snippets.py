"""
This module contains the code snippets found in Chapter 10 of "Advances in Financial Machine Learning" by
Marcos López de Prado. The code has been amended for readability, to conform to PEP8 rules, to keep the snippets as
manageable single-units of functionality, as well as to account for deprecation of functions originally used, but is
otherwise unaltered.
"""

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
    # Get signals from predictions.
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) Generate signals from multinomial classification (one-vs-rest).
    signal0 = (prob - 1/num_classes) / (prob * (1 - prob))**0.5

    # Allow for bet size to be returned with or without side.
    if not isinstance(pred, type(None)):
        # signal = side * size
        signal0 = pred * (2 * norm.cdf(signal0) - 1)
    else:
        # signal = size only
        signal0 = signal0.apply(lambda s: 2 * norm.cdf(s) - 1)

    # Note 1: In the book, this function contains a conditional statement checking for a column named 'side',
    # then executes what is essentially the above line. This has been removed as it appears to be redundant
    # and simplifies the function.

    # Note 2: In the book, this function includes the averaging and discretization steps, which are omitted here.
    # The functions for performing these are included in this file, and can be applied as options in the user-level
    # functions in bet_sizing.py.

    return signal0


def avg_active_signals(signals, num_threads=1):
    """
    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
    Function averages the bet sizes of all concurrently active bets. This function makes use of multiprocessing.

    :param signals: (pandas.DataFrame) Contains at least the following columns:
        'signal' - the bet size
        't1' - the closing time of the bet
    :param num_threads: (int) Number of threads to use in multiprocessing, default value is 1.
    :return: (pandas.Series) The averaged bet sizes.
    """

    # 1) Time points where signals change (either one start or one ends).
    t_pnts = set(signals['t1'].dropna().to_numpy())
    t_pnts = t_pnts.union(signals.index.to_numpy())
    t_pnts = list(t_pnts)
    t_pnts.sort()
    out = mp_pandas_obj(mp_avg_active_signals, ('molecule', t_pnts), num_threads, signals=signals)
    return out


def mp_avg_active_signals(signals, molecule):
    """
    Part of SNIPPET 10.2
    A function to be passed to the 'mp_pandas_obj' function to allow the bet sizes to be averaged using multiprocessing.

    At time loc, average signal among those still active.
    Signal is active if:
        a) it is issued before or at loc, and
        b) loc is before the signal's end time, or end time is still
            unknown (NaT).

    :param signals: (pandas.DataFrame) Contains at least the following columns:
        'signal' - the bet size
        't1' - the closing time of the bet
    :param molecule: (list) Indivisible tasks to be passed to 'mp_pandas_obj', in this case a list of datetimes.
    :return: (pandas.Series) The averaged bet size sub-series.
    """
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.to_numpy() <= loc)&((loc < signals['t1'])|pd.isnull(signals['t1']))
        act = signals[df0].index
        if act.size > 0:
            # Average active signals if they exist.
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            # Return zero if no signals are active at this time step.
            out[loc] = 0
    return out


def discrete_signal(signal0, step_size):
    """
    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING
    Discretizes the bet size signal based on the step size given.

    :param signal0: (pandas.Series) The signal to discretize.
    :param step_size: (float) Step size.
    :return: (pandas.Series) The discretized signal.
    """
    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1  # Cap
    signal1[signal1 < -1] = -1  # Floor
    return signal1

# ==============================================================================
# SNIPPET 10.4 - DYNAMIC POSITION SIZE AND LIMIT PRICE
# The below functions are part of or derived from the functions
# in snippet 10.4.
# ==============================================================================
# Bet size calculations based on a sigmoid function.
def bet_size_sigmoid(w, x):
    """
    Part of SNIPPET 10.4
    Calculates the bet size from the price divergence and
    a regulating coefficient.

    :param w: (float) Coefficient regulating the width of the bet size function.
    :param x: (float) Price divergence, forecast price - market price.
    :return: (float) The bet size.
    """
    return x * ((w + x**2)**(-0.5))

def get_t_pos_sigmoid(w, f, m_p, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the target position given the forecast price, market price,
    maximum position size, and a regulating coefficient.

    :param w: (float) Coefficient regulating the width of the bet size function.
    :param f: (float) Forecast price.
    :param m_p: (float) Market price.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Target position.
    """
    return int(bet_size_sigmoid(w, f-m_p) * max_pos)

def inv_price_sigmoid(f, w, m):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.

    :param f: (float) Forecast price.
    :param w: (float) Coefficient regulating the width of the bet size function.
    :param m: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """
    return f - m * (w/(1/m**2))**(0.5)

def limit_price_sigmoid(t_pos, pos, f, w, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the limit price.

    :param t_pos: (float) Target position.
    :param pos: (float) Current position.
    :param f: (float) Forecast price.
    :param w: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Limit price.
    """
    sgn = np.sign(t_pos-pos)
    l_p = 0
    for j in range(abs(pos+sgn), abs(t_pos+1)):
        #print(j, inv_price_sigmoid(f, w, j/float(max_pos)))
        l_p += inv_price_sigmoid(f, w, j/float(max_pos))
    l_p = l_p / abs(t_pos-pos)
    return l_p

def get_w_sigmoid(x, m):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the
    regulating coefficient 'w'.

    :param x: (float) Price divergence, forecast price - market price.
    :param m: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """
    return (x**2) * ((m**(-2)) - 1)

# ==============================================================================
# Bet size calculation functions, power and sigmoid packaged together.
# This is useful as more bet sizing function options are added.
def bet_size(w, x, func):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and
    a regulating coefficient.

    :param w: (float) Coefficient regulating the width of the bet size function.
    :param x: (float) Price divergence, f - m_p
    :param func: (string) Function to use for dynamic calculation. Valid
        options are: 'sigmoid'.
    :return: (float) The bet size.
    """
    if func == 'sigmoid':
        return bet_size_sigmoid(w, x)
    else:
        raise ValueError("Argument 'func' must be one of: 'sigmoid', 'power'")

def get_t_pos(w, f, m_p, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price,
    maximum position size, and a regulating coefficient.

    :param w: (float) Coefficient regulating the width of the bet size function.
    :param f: (float) Forecast price.
    :param m_p: (float) Market price.
    :param max_pos: (float) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid
        options are: 'sigmoid'.
    :return: (float) Target position.
    """
    if func == 'sigmoid':
        return get_t_pos_sigmoid(w, f, m_p, max_pos)
    else:
        raise ValueError("Argument 'func' must be one of: 'sigmoid', 'power'")

def inv_price(f, w, m, func):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price.

    :param t_pos: (float) Target position.
    :param pos: (float) Current position.
    :param f: (float) Forecast price.
    :param w: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (float) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid
        options are: 'sigmoid'.
    :return: (float) Limit price.
    """
    if func == 'sigmoid':
        return inv_price_sigmoid(f, w, m)
    else:
        raise ValueError("Argument 'func' must be one of: 'sigmoid', 'power'")

def limit_price(t_pos, pos, f, w, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the
    regulating coefficient 'w'.

    :param x: (float) Price divergence, forecast price - market price.
    :param m: (float) Bet size.
    :param func: (string) Function to use for dynamic calculation. Valid
        options are: 'sigmoid'.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """
    if func == 'sigmoid':
        return limit_price_sigmoid(t_pos, pos, f, w, max_pos)
    else:
        raise ValueError("Argument 'func' must be one of: 'sigmoid', 'power'")

def get_w(x, m, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the
    regulating coefficient 'w'.

    :param x: (float) Price divergence, forecast price - market price.
    :param m: (float) Bet size.
    :param func: (string) Function to use for dynamic calculation. Valid
        options are: 'sigmoid'.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """
    if func == 'sigmoid':
        return get_w_sigmoid(x, m)
    else:
        raise ValueError("Argument 'func' must be one of: 'sigmoid', 'power'")
