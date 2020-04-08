import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


# General OLPS class
class OLPS(object):
    # Initialize
    def __init__(self):
        """
        Constructor
        """
        self.weights = list()
        self.asset_prices = None
        self.covariance_matrix = None
        self.portfolio_risk = None
        self.portfolio_return = None
        self.portfolio_sharpe_ratio = None
        self.expected_returns = None
        self.returns_estimator = ReturnsEstimation()

    # allocate weights
    def allocate(self,
                 asset_names,
                 asset_prices=None,
                 covariance_matrix=None,
                 expected_asset_returns=None,
                 risk_free_rate=0.05,
                 resample_by=None):
        """
        :param asset_names: (list) a list of strings containing the asset names
        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param expected_asset_returns: (list/np.array/pd.dataframe) a list of mean stock returns (mu)
        :param covariance_matrix: (pd.Dataframe/numpy matrix) user supplied covariance matrix of asset returns (sigma)
        :param risk_free_rate: (float) the rate of return for a risk-free asset.
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        self.initial_check(asset_prices, expected_asset_returns, covariance_matrix)

    def initial_check(asset_prices, expected_asset_returns, covariance_matrix):
        if asset_prices is None and (expected_asset_returns is None or covariance_matrix is None):
            raise ValueError("Either supply your own asset returns matrix or pass the asset prices as input")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")