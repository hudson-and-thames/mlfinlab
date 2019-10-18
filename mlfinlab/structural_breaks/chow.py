"""
Explosivness tests: Chow-Type Dickey-Fuller Test
"""

import pandas as pd
from mlfinlab.structural_breaks.sadf import _get_betas
from mlfinlab.util import mp_pandas_obj


def _get_dfc_for_t(df, molecule):
    """
    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule
    """

    dfc_series = pd.Series(index=molecule)

    for index in molecule:
        df_diff = df.diff().dropna()
        df_shifted_diff = df.shift(1).dropna()
        df_shifted_diff[:index] = 0  # D_t* indicator: before t* D_t* = 0

        y = df_diff.loc[df_shifted_diff.index].values
        x = df_shifted_diff.values
        coefs, coef_vars = _get_betas(y, x.reshape(-1, 1))
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series[index] = b_estimate / (b_var ** 0.5)
    return dfc_series


def get_chow_type_stat(df, min_length, num_threads=8):
    """
    Multithread implementation of Chow-Type Dickey-Fuller Test
    """
    molecule = df.index[
               min_length:df.shape[0] - min_length]  # Indices to test. We drop min_length first and last values
    dfc_series = mp_pandas_obj(func=_get_dfc_for_t,
                               pd_obj=('molecule', molecule),
                               df=df,
                               num_threads=num_threads,
                               )
    return dfc_series
