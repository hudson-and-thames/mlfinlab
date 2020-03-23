"""
Logic regarding concurrent labels from chapter 4.
"""

import pandas as pd

from mlfinlab.util.multiprocess import mp_pandas_obj


def num_concurrent_events(close_series_index, label_endtime, molecule):
    """
    Snippet 4.1, page 60, Estimating the Uniqueness of a Label
    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param close_series_index: (pd.Series) close prices index
    :param label_endtime: (pd.Series) label endtime series (t1 for triple barrier events)
    :param molecule: (an array) a set of datetime index values for processing
    :return: (pd.Series) number concurrent labels for each datetime index
    """
    # Find events that span the period [molecule[0], molecule[1]]
    label_endtime = label_endtime.fillna(close_series_index[-1])  # Unclosed events still must impact other weights
    label_endtime = label_endtime[label_endtime >= molecule[0]]  # Events that end at or after molecule[0]
    # Events that start at or before t1[molecule].max()
    label_endtime = label_endtime.loc[:label_endtime[molecule].max()]

    # Count events spanning a bar
    nearest_index = close_series_index.searchsorted(pd.DatetimeIndex([label_endtime.index[0], label_endtime.max()]))
    count = pd.Series(0, index=close_series_index[nearest_index[0]:nearest_index[1] + 1])
    for t_in, t_out in label_endtime.iteritems():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0]:label_endtime[molecule].max()]


def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    """
    Snippet 4.2, page 62, Estimating the Average Uniqueness of a Label
    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param label_endtime: (pd.Series) label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (an array) a set of datetime index values for processing.
    :return: (pd.Series) average uniqueness over event's lifespan.
    """
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for t_in, t_out in label_endtime.loc[wght.index].iteritems():
        wght.loc[t_in] = (1. / num_conc_events.loc[t_in:t_out]).mean()
    return wght


def get_av_uniqueness_from_triple_barrier(triple_barrier_events, close_series, num_threads):
    """
    This function is the orchestrator to derive average sample uniqueness from a dataset labeled by the triple barrier
    method.

    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param close_series: (pd.Series) close prices.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :return: (pd.Series) average uniqueness over event's lifespan for each index in triple_barrier_events
    """
    out = pd.DataFrame()
    num_conc_events = mp_pandas_obj(num_concurrent_events, ('molecule', triple_barrier_events.index), num_threads,
                                    close_series_index=close_series.index, label_endtime=triple_barrier_events['t1'])
    num_conc_events = num_conc_events.loc[~num_conc_events.index.duplicated(keep='last')]
    num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)
    out['tW'] = mp_pandas_obj(_get_average_uniqueness, ('molecule', triple_barrier_events.index), num_threads,
                              label_endtime=triple_barrier_events['t1'], num_conc_events=num_conc_events)
    return out
