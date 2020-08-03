"""
Logic regarding concurrent labels from chapter 4.
"""

import pandas as pd

from mlfinlab.util.multiprocess import mp_pandas_obj


def num_concurrent_events(close_series_index, label_endtime, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.1, page 60.

    Estimating the Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param close_series_index: (pd.Series) Close prices index
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.Series) Number concurrent labels for each datetime index
    """

    pass


def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.2, page 62.

    Estimating the Average Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Average uniqueness over event's lifespan.
    """

    pass


def get_av_uniqueness_from_triple_barrier(triple_barrier_events, close_series, num_threads, verbose=True):
    """
    This function is the orchestrator to derive average sample uniqueness from a dataset labeled by the triple barrier
    method.

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Average uniqueness over event's lifespan for each index in triple_barrier_events
    """

    pass
