"""
Logic regarding return and time decay attribution for sample weights from chapter 4.
"""

import numpy as np
import pandas as pd

from mlfinlab.sampling.concurrent import num_concurrent_events, get_av_uniqueness_from_triple_barrier
from mlfinlab.util.multiprocess import mp_pandas_obj


def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.10, page 69.

    Determination of Sample Weight by Absolute Return Attribution

    Derives sample weights based on concurrency and return. Works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) Close prices
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Sample weights based on number return and concurrency for molecule
    """

    pass

def get_weights_by_return(triple_barrier_events, close_series, num_threads=5, verbose=True):
    """
    Advances in Financial Machine Learning, Snippet 4.10(part 2), page 69.

    Determination of Sample Weight by Absolute Return Attribution

    This function is orchestrator for generating sample weights based on return using mp_pandas_obj.

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Sample weights based on number return and concurrency
    """

    pass


def get_weights_by_time_decay(triple_barrier_events, close_series, num_threads=5, decay=1, verbose=True):
    """
    Advances in Financial Machine Learning, Snippet 4.11, page 70.

    Implementation of Time Decay Factors

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param decay: (int) Decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old
        - decay = 0 means that weights converge linearly to zero, as they become older
        - decay < 0 means that the oldes portion c of the observations receive zero weight (i.e they are erased from memory)
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Sample weights based on time decay factors
    """

    pass
