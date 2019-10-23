"""
Explosivness tests: SADF
"""

import pandas as pd
import numpy as np
from mlfinlab.util.multiprocess import mp_pandas_obj


def _get_sadf_at_t(X, y, min_length):
    """
    Snippet 17.2, page 258. SADF's Inner Loop (get sadf value at t)
    """
    start_points, bsadf = range(0, y.shape[0] - min_length + 1), -np.inf
    for start in start_points:
        y_, x_ = y[start:], X[start:]
        b_mean_, b_std_ = _get_betas(y_, x_)
        if not np.isnan(b_mean_[0]):
            b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
            all_adf = b_mean_ / b_std_
            if all_adf > bsadf:
                bsadf = all_adf
    return bsadf


def _get_y_x(series, model, lags, add_const):
    """
    Snippet 17.2, page 258-259. Preparing The Datasets
    """
    series = pd.DataFrame(series)
    series_diff = series.diff().dropna()
    x = _lag_df(series_diff, lags).dropna()
    x['y_lagged'] = series.shift(1).loc[x.index]  # add y_(t-1) column
    y = series_diff.loc[x.index]

    if add_const is True:
        x['const'] = 1
    if model == 'linear':
        x['trend'] = np.arange(x.shape[0])  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
    elif model == 'quadratic':
        x['trend'] = np.arange(x.shape[0]) ** 2  # Add t^2 to the model (0, 1, 4, 9, ....)

    # Move y_lagged column to the front for further extraction
    columns = list(x.columns)
    columns.insert(0, columns.pop(columns.index('y_lagged')))
    x = x[columns]
    return x, y


def _lag_df(df, lags):
    """
    Snipet 17.3, page 259. Apply Lags to DataFrame
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + '_' + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how='outer')
    return df_lagged


def _get_betas(y, x):
    """
    Snippet 17.4, page 259. Fitting The ADF Specification
    """
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    try:
        xx_inv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return [np.nan], [[np.nan, np.nan]]
    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(x, b_mean)
    b_var = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xx_inv
    return b_mean, b_var


def _sadf_outer_loop(X, y, min_length, molecule):
    """
    This function gets SADF for t times from molecule
    """
    sadf_series = pd.Series(index=molecule)
    for index in molecule:
        X_subset = X.loc[:index].values
        y_subset = y.loc[:index].values.reshape(-1, 1)
        value = _get_sadf_at_t(X_subset, y_subset, min_length)
        sadf_series[index] = value
    return sadf_series


def get_sadf(df, min_length, model, add_const, lags, num_threads=8):
    """
    Multithread implementation of SADF
    """
    X, y = _get_y_x(df, model, lags, add_const)
    molecule = y.index[min_length:y.shape[0]]

    sadf_series = mp_pandas_obj(func=_sadf_outer_loop,
                                pd_obj=('molecule', molecule),
                                X=X,
                                y=y,
                                min_length=min_length,
                                num_threads=num_threads,
                                )
    return sadf_series


df = pd.read_csv('../../../../Downloads/pnl_and_commissions.csv', index_col=0, parse_dates=[0])
df['equity'] = 1e6 + df.realised_pnl + df.unrealised_pnl
sadf = get_sadf(np.log(df.equity), min_length=20, model='linear', add_const=True, lags=6)
