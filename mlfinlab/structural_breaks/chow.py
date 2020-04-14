"""
Explosiveness tests: Chow-Type Dickey-Fuller Test
"""

import pandas as pd
from mlfinlab.structural_breaks.sadf import get_betas
from mlfinlab.util import mp_pandas_obj


# pylint: disable=invalid-name

def _get_dfc_for_t(series: pd.Series, molecule: list) -> pd.Series:
    """
    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule

    :param series: (pd.Series) series to test
    :param molecule: (list) of dates to test
    :return: (pd.Series) fo statistics for each index from molecule
    """

    dfc_series = pd.Series(index=molecule)

    for index in molecule:
        series_diff = series.diff().dropna()
        series_lag = series.shift(1).dropna()
        series_lag[:index] = 0  # D_t* indicator: before t* D_t* = 0

        y = series_diff.loc[series_lag.index].values
        x = series_lag.values
        coefs, coef_vars = get_betas(x.reshape(-1, 1), y)
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series[index] = b_estimate / (b_var ** 0.5)

    return dfc_series


def get_chow_type_stat(series: pd.Series, min_length: int = 20, num_threads: int = 8) -> pd.Series:
    """
    Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252

    :param series: (pd.Series) series to test
    :param min_length: (int) minimum sample length used to estimate statistics
    :param num_threads: (int): number of cores to use
    :return: (pd.Series) of Chow-Type Dickey-Fuller Test statistics
    """
    # Indices to test. We drop min_length first and last values
    molecule = series.index[min_length:series.shape[0] - min_length]
    dfc_series = mp_pandas_obj(func=_get_dfc_for_t,
                               pd_obj=('molecule', molecule),
                               series=series,
                               num_threads=num_threads,
                               )
    return dfc_series
