"""
Implementation of Chu-Stinchcombe-White test
"""

import pandas as pd
import numpy as np
from mlfinlab.util import mp_pandas_obj


def _get_values_diff(test_type, series, index, ind):
    """
    Gets the difference between two values given a test type.
    :param test_type: one_sided or two_sided
    :param series: Series of values
    :param index: primary index
    :param ind: secondary index
    :return: Difference between 2 values
    """
    if test_type == 'one_sided':
        values_diff = series.loc[index] - series.loc[ind]
    elif test_type == 'two_sided':
        values_diff = abs(series.loc[index] - series.loc[ind])
    else:
        raise ValueError('Test type is unknown: can be either one_sided or two_sided')

    return values_diff


def _get_s_n_for_t(series: pd.Series, test_type: str, molecule: list) -> pd.Series:
    """
    Get maximum S_n_t value for each value from molecule for Chu-Stinchcombe-White test
    :param series: (pd.Series) to get statistics for
    :param test_type: (str): two-sided or one-sided test
    :param molecule: (list) of indices to get test statistics for
    :return: (pd.Series) of statistics
    """

    s_n_t_series = pd.DataFrame(index=molecule, columns=['stat', 'critical_value'])
    for index in molecule:

        series_t = series.loc[:index]
        squared_diff = series_t.diff().dropna() ** 2
        integer_index = series_t.index.get_loc(index)
        sigma_sq_t = 1 / (integer_index - 1) * sum(squared_diff)

        max_s_n_value = -np.inf
        max_s_n_critical_value = None  # Corresponds to c_alpha[n,t]

        # Indices difference for the last index would yield 0 -> division by zero warning,
        # no need to index last value iteration
        for ind in series_t.index[:-1]:
            values_diff = _get_values_diff(test_type, series, index, ind)
            temp_integer_index = series_t.index.get_loc(ind)
            s_n_t = 1 / (sigma_sq_t * np.sqrt(integer_index - temp_integer_index)) * values_diff
            if s_n_t > max_s_n_value:
                max_s_n_value = s_n_t
                max_s_n_critical_value = np.sqrt(
                    4.6 + np.log(integer_index - temp_integer_index))  # 4.6 is b_a estimate derived via Monte-Carlo

        s_n_t_series.loc[index, ['stat', 'critical_value']] = max_s_n_value, max_s_n_critical_value
    return s_n_t_series


def get_chu_stinchcombe_white_statistics(series: pd.Series, test_type: str = 'one_sided',
                                         num_threads: int = 8) -> pd.Series:
    """
    Multithread Chu-Stinchcombe-White test implementation, p.251

    :param series: (pd.Series) to get statistics for
    :param test_type: (str): two-sided or one-sided test
    :param num_threads: (int) number of cores
    :return: (pd.Series) of statistics
    """
    molecule = series.index[2:series.shape[0]]  # For the first two values we don't have enough info

    s_n_t_series = mp_pandas_obj(func=_get_s_n_for_t,
                                 pd_obj=('molecule', molecule),
                                 series=series,
                                 test_type=test_type,
                                 num_threads=num_threads,
                                 )
    return s_n_t_series
