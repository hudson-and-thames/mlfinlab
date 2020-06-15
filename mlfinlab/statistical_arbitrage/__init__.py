"""
Module for Statistical Arbitrage.
"""

from mlfinlab.statistical_arbitrage.stationarity import calc_adfuller, calc_kpss
from mlfinlab.statistical_arbitrage.cointegration import calc_engle_granger, calc_johansen
from mlfinlab.statistical_arbitrage.regression import calc_rolling_regression, calc_all_regression
