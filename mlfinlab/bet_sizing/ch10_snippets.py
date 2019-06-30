"""
This module contains the code snippets found in Chapter 10 of
"Advances in Financial Machine Learning" by Marcos López de Prado.
The code has been ammended for readability, to conform to PEP8
rules, to keep the snippets as manageable single-units of
functionality, as well as to account for depracation of functions
originally used, but is otherwise unaltered.
"""


# imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from mlfinlab.util.multiprocess import mp_pandas_obj


def get_signal(events, prob, pred, num_classes):
    """
    SNIPPET 10.1 - FROM PROBABILITIES TO BET SIZE
    
    :param events: (pandas.DataFrame)
    :param prob: (???)
    :param pred: (???)
    :param num_classes: (int)
    :return: (pd.Series)
    """
    # get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) generate signals from multinomial classification (one-vs-rest)
    signal0 = (prob - 1/num_classes) / (prob * (1 - prob))**0.5
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal = side * size

    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']  # meta-labeling
    
    # Note: In the book, this function includes the averaging and
    # discretization steps, which are omitted here. The functions
    # for performing these are included in this file, and can be
    # applied as options in the user-level functions in bet_sizing.py.

    return signal0


def avg_active_signals(signals, num_threads=1):
    """
    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
    Function averages the bet sizes of all concurrently active bets.
    This function makes use of multiprocessing.
    
    :param signals: (pandas.Series) The bet sizes.
    :param num_threads: (int) Number of threads to use in multiprocessing,
        default value is 1.
    :return: (pandas.Series) The averaged bet sizes.
    """
    # 1) time points where signals change (either one start or one ends)
    t_pnts = set(signals['t1'].dropna().to_numpy())
    t_pnts = t_pnts.union(signals.index.to_numpy())
    t_pnts = list(t_pnts)
    t_pnts.sort()
    out = mp_pandas_obj(mp_avg_active_signals, ('molecule', t_pnts),
                        num_threads, signals=signals)
    return out


def mp_avg_active_signals(signals, molecule):
    """
    Part of SNIPPET 10.2

    A function to be passed to the 'mp_pandas_obj' function to allow the
    bet sizes to be averaged using multiprocessing.

    At time loc, average signal among those still active.
    Signal is active if:
        a) it is issued before or at loc, and
        b) loc is before the signal's end time, or end time is still
            unknown (NaT).

    :param signals: (list)
    :param molecule: 
    :return: (pandas.DataFrame)
    """
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.to_numpy() <= loc)&\
            ((loc < signals['t1'])|pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            # average active signals
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            # no signals active at this time
            out[loc] = 0
    return out


def discrete_signal(signal0, step_size):
    """
    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING

    Discretizes the bet size signal based on the step size given.

    :param signal0: (pandas.Series)
    :param step_size: (float) Step size between (0, 1].
    :return: (pandas.Series)
    """
    signal1 = (signal0 / step_size).round()*step_size
    signal1[signal1>1] = 1  # cap
    signal1[signal1<-1] = -1  # floor
    return signal1
