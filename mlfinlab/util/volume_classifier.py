from scipy.stats import norm
import numpy as np


def get_bvc_buy_volume(close, volume, window=20):
    return volume * norm.cdf(close.diff() / close.diff().rolling(window=window).std())


def get_tick_rule(price, prev_price):
    if price > prev_price:
        return 1
    elif price < prev_price:
        return -1
    return np.nan
