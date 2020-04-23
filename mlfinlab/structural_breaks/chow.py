"""
Explosiveness tests: Chow-Type Dickey-Fuller Test
"""

import pandas as pd
import numpy as np
from numba import njit


# pylint: disable=invalid-name

@njit
def _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values):
    """
    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule
    :param molecule_range: (np.array) of dates to test
    :param series_lag_values_start: (int) offset series because of min_length
    :return: (pd.Series) fo statistics for each index from molecule
    """
    dfc_series = []
    for i in molecule_range:
        ### TEST
        # index = molecule[0]
        ### TEST
        series_lag_values_ = series_lag_values.copy()
        series_lag_values_[:(series_lag_values_start + i)] = 0  # D_t* indicator: before t* D_t* = 0

        # define x and y for regression
        y = series_diff
        x = series_lag_values_.reshape(-1, 1)
        
        # Get regression coefficients estimates
        xy = x.transpose() @ y
        xx = x.transpose() @ x

        # calculate to check for singularity
        det = np.linalg.det(xx)

        # get coefficient and std from linear regression
        if det == 0:
            b_mean = [np.nan]
            b_std = [[np.nan, np.nan]]
        else:
            xx_inv = np.linalg.inv(xx)
            coefs = xx_inv @ xy
            err = y - (x @ coefs)
            coef_vars = np.dot(np.transpose(err), err) / (x.shape[0] - x.shape[1]) * xx_inv
            
        b_estimate, b_var = coefs[0], coef_vars[0][0]
        dfc_series.append(b_estimate / (b_var ** 0.5))
        
    return dfc_series


def get_chow_type_stat(series: pd.Series, min_length: int = 20) -> pd.Series:
    """
    Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252
    :param series: (pd.Series) series to test
    :param min_length: (int) minimum sample length used to estimate statistics
    :param num_threads: (int): number of cores to use
    :return: (pd.Series) of Chow-Type Dickey-Fuller Test statistics
    """
    # Indices to test. We drop min_length first and last values
    molecule = series.index[min_length:series.shape[0] - min_length]
    molecule = molecule.values
    molecule_range = np.arange(0, len(molecule))

    series_diff = series.diff().dropna()
    series_diff = series_diff.values
    series_lag = series.shift(1).dropna()
    series_lag_values = series_lag.values
    series_lag_times_ = series_lag.index.values
    series_lag_values_start = np.where(series_lag_times_ == molecule[0])[0].item() + 1
    
    dfc_series = _get_dfc_for_t(molecule_range, series_lag_values_start, series_diff, series_lag_values)
    
    dfc_series = pd.Series(dfc_series, index=molecule)
    
    return dfc_series
