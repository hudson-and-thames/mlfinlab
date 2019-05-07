"""
Logic regarding return and time decay attribution for sample weights from chapter 4.
"""

import pandas as pd
import numpy as np
from mlfinlab.util.multiprocess import mp_pandas_obj


def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    """
    Snippet 4.10, page 69, Determination of Sample Weight by Absolute Return Attribution
    Derives sample weights based on concurrency and return. Works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) close prices
    :param molecule: (an array) a set of datetime index values for processing.
    :return: (pd.Series) of sample weights based on number return and concurrency for molecule
    """

    ret = np.log(close_series).diff()  # log-returns, so that they are additive
    weights = pd.Series(index=molecule)
    for t_in, t_out in label_endtime.loc[weights.index].iteritems():
        weights.loc[t_in] = (ret.loc[t_in:t_out] / num_conc_events.loc[
                                                  t_in:t_out]) / sum()  # weights depend on returns and label concurrency
    return weights.abs()


def get_weights_by_return(triple_barrier_events, num_conc_events, close_series, molecule, num_threads):
    """
    Snippet 4.10(part 2), page 69, Determination of Sample Weight by Absolute Return Attribution
    This function is orchestrator for generating sample weights based on return using mp_pandas_obj.
    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param num_conc_events: (pd.Series) number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) close prices
    :param molecule: (an array) a set of datetime index values for processing.
    :param num_threads: (int) the number of threads concurrently used by the function.
    :return: (pd.Series) of sample weights based on number return and concurrency
    """
    weights = mp_pandas_obj(_apply_weight_by_return, ('molecule', triple_barrier_events.index), num_threads,
                            label_endtime=triple_barrier_events['t1'], num_conc_events=num_conc_events,
                            close_series=close_series)
    weights *= weights.shape[0] / weights.sum()
    return weights


def get_weights_by_time_decay(av_uniqueness, decay=1):
    """
    Snippet 4.11, page 70, Implementation of Time Decay Factors
    :param av_uniqueness: (pd.Series) average uniqueness over events from triple barrier method,  result of get_av_uniqueness_from_tripple_barrier
    :param decay: (int) decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old
        - decay = 0 means that weights converge linearly to zero, as they become older
        - decay < 0 means that the oldes portion c of the observations receive zero weight (i.e they are erased from memory)
    :return: (pd.Series) of sample weights based on time decay factors
    """
    # apply piecewise-linear decay to observed uniqueness
    # newest observation gets weight=1, oldest observation gets weight=decay
    decay_w = av_uniqueness['tW'].sort_index().cumsum()
    if decay >= 0:
        slope = (1 - decay) / decay_w.iloc[-1]
    const = 1 - slope * decay_w.iloc[-1]
    decay_w = const + slope * decay_w
    decay_w[decay_w < 0] = 0  # weights can't be negative
    return decay_w
