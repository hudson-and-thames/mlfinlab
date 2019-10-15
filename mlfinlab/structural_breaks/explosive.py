def get_bsadf(log_prices, min_length, constant, lags):
    """
    Snippet 17.2, page 258. SADF's Inner Loop
    """
    y, x = _get_y_x(log_prices, constant=constant, lags=lags)
    start_points, bsadf, all_adf = range(0, y.shape[0]+lags-min_length+1), None, []
    for start in start_points:
        y_, x_ = y[start:], x[start:]
        b_mean_, b_std_ = _get_betas(y_, x_)
        b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0]**0.5
        all_adf.append(b_mean_/b_std_)
        if all_adf[-1] > bsadf:
            bsadf = all_adf[-1]
    out = {'Time': log_prices.index[-1], 'gsadf': bsadf}
    return out

def _get_y_x(series, constant, lags):
    """
    Snippet 17.2, page 258-259. Preparing The Datasets
    """
    series_diff = series.diff().dropna()
    x = _lag_df(series_diff, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0] # lagged level
    y = series_diff.iloc[-x.shape[0]:].values

    if constant != 'const':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'linear':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'quadratic':
            x = np.append(x, trend**2, axis=1)
    return y, x

def _lag_df(df, lags):
    """
    Snipet 17.3, page 259. Apply Lags to DataFrame
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags+1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + '_' + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how = 'outer')
    return df_lagged

def _get_betas(y, x):
    """
    Snippet 17.4, page 259. Fitting The ADF Specidication
    """
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xx_inv = np.linalg.inv(xx)
    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(x, b_mean)
    b_var = np.dot(err.T, err) / (x.shape[0] - x.shape[1])*xx_inv
    return b_mean, b_var
