"""
Module for Statistical Arbitrage.
"""

# Base class.
from mlfinlab.statistical_arbitrage.base import StatArb

# Eigenportfolio.
from mlfinlab.statistical_arbitrage.eigenportfolio import Eigenportfolio

# Signals.
from mlfinlab.statistical_arbitrage.signals import calc_zscore, calc_ou_process

# Stationarity.
from mlfinlab.statistical_arbitrage.stationarity import calc_adfuller, calc_kpss

# Cointegration.
from mlfinlab.statistical_arbitrage.cointegration import calc_engle_granger, calc_johansen
