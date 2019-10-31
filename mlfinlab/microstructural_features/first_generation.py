import numpy as np
import pandas as pd


def get_roll_measure(close_prices, window):
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))


def get_roll_impact(close_prices, dollar_volume, window):
    roll_measure = get_roll_measure(close_prices, window)
    return roll_measure / dollar_volume


# Corwin-Schultz algorithm
def _get_beta(high, low, sample_length):
    ret = np.log(high / low)
    hl = ret ** 2
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sample_length).mean()
    return beta.dropna()


def _get_gamma(high, low):
    h2 = high.rolling(window=2).max()
    l2 = low.rolling(window=2).min()
    gamma = np.log(h2 / l2) ** 2
    return gamma.dropna()


def _get_alpha(beta, gamma):
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha.dropna()


def get_corwin_schultz_estimator(high, low, sample_length=20):
    # Note: S<0 iif alpha<0
    beta = _get_beta(high, low, sample_length)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread


def get_bekker_parkinson_vol(high, low, sample_length=20):
    beta = _get_beta(high, low, sample_length)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma
