import pandas as pd
from sklearn.linear_model import LinearRegression

def get_bar_based_kyle_lambda(close, volume, window=20):
    close_diff = close.diff().dropna(inplace=True)
    close_diff_sign = np.sign(close_diff)
    volume_mult_trade_signs = volume * close_diff_sign
    volume_mult_trade_signs = volume_mult_trade_signs.loc[close_diff.index] # bt * Vt
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()

def get_bar_based_amihud_lambda(close, dollar_volume, window=20):
    returns_abs = np.log(close/close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()

def get_bar_based_hasbrouck_lambda(close, dollar_volume, window=20):
    log_ret = np.log(close / close.shift(1))
    log_ret_sign = np.sign(log_ret)

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()

def get_trades_based_kyle_lambda(price_diff, volume, aggressor_flags):
    model = LinearRegression(fit_intercept=False, copy_X=False)
    signed_volume = volume * aggressor_flags
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    model.fit(X, y)
    return model.coef_[0]

def get_trades_based_amihud_lambda(log_ret, dollar_volume):
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]

def get_trades_based_hasbrouck_lambda(log_ret, dollar_volume, aggressor_flags):
    model = LinearRegression(fit_intercept=False, copy_X=False)
    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    model.fit(X, y)
    return model.coef_[0]
