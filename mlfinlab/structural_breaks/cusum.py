"""
Implementation of Chu-Stinchcombe-White test
"""

import pandas as pd
import numpy as np
from mlfinlab.util import mp_pandas_obj

def _get_s_n_for_t(df, type, molecule):
    """
    Get maximum S_n_t value for each value from molecule for Chu-Stinchcombe-White test
    """

    s_n_t_series = pd.DataFrame(index=molecule, columns = ['stat', 'critical_value'])
    for index in molecule:

        df_t = df.loc[:index]
        squared_diff = df_t.diff().dropna()**2
        integer_index = df_t.index.get_loc(index)
        sigma_sq_t = 1/(integer_index-1) * sum(squared_diff)

        max_s_n_value = -np.inf
        max_s_n_critical_value = None # Corresponds to c_aplha[n,t]

        # Indices difference for the last index would yield 0 -> division by zero warning, no need to index last value iteration
        for ind in df_t.index[:-1]:

            if type == 'one_sided':
                values_diff = df.loc[index] - df.loc[ind]
            elif type == 'two_sided':
                values_diff = abs(df.loc[index] - df.loc[ind])
            else:
                raise ValueError('Test type is unknown: can be either one_sided or two_sided')

            temp_integer_index = df_t.index.get_loc(ind)
            s_n_t = 1/(sigma_sq_t * np.sqrt(integer_index-temp_integer_index)) * (values_diff)
            if s_n_t > max_s_n_value:
                max_s_n_value = s_n_t
                max_s_n_critical_value = np.sqrt(4.6 + np.log(integer_index-temp_integer_index))

        s_n_t_series.loc[index, ['stat', 'critical_value']] = max_s_n_value, max_s_n_critical_value
    return s_n_t_series

def get_chu_stinchcombe_white_statistics(df, type='one_sided', num_threads=8):
    """
    Multithread Chu-Stinchcombe-White test implementation
    """
    molecule = df.index[2:df.shape[0]] # For the first two values we don't have enough info

    s_n_t_series = mp_pandas_obj(func=_get_s_n_for_t,
                               pd_obj=('molecule', molecule),
                               df=df,
                               type=type,
                               num_threads=num_threads,
                               )
    return s_n_t_series
