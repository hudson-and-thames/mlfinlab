import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS

from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class BAH(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    The Buy and Hold strategy one invests wealth among the market with an initial portfolio of weights and holds
    the portfolio till the end. The manager only buys the assets at the beginning of the first period and does
    not rebalance in subsequent periods.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

    def allocate(self,
                 asset_names,
                 asset_prices,
                 covariance_matrix=None,
                 expected_asset_returns=None,
                 risk_free_rate=0.05,
                 weights=None,
                 resample_by=None):
        """
        :param asset_names: (list) a list of strings containing the asset names
        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param expected_asset_returns: (list/np.array/pd.dataframe) a list of mean stock returns (mu)
        :param covariance_matrix: (pd.Dataframe/numpy matrix) user supplied covariance matrix of asset returns (sigma)
        :param risk_free_rate: (float) the rate of return for a risk-free asset.
        :param weights: (list) a list of weights, if not stated weights are uniform
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        # initial check
        # initial_check(asset_prices, expected_asset_returns, covariance_matrix)

        number_of_assets = len(asset_names)
        if weights is None:
            self.weights = np.linspace(0, 1, num=number_of_assets)
        else:
            self.weights = weights

        # Calculate covariance of returns or use the user specified covariance matrix
        covariance_matrix = calculate_covariance(asset_names, asset_prices, covariance_matrix, resample_by, self.returns_estimator)

        # Calculate the expected returns if the user does not supply any returns
        # expected_asset_returns = calculate_expected_asset_returns(asset_prices, expected_asset_returns, resample_by)

        # Calculate the portfolio risk and return if it has not been calculated
        # self.portfolio_risk = calculate_portfolio_risk(self.portfolio_risk, covariance_matrix, self.weights)

        # Calculate the portfolio return
        # self.portfolio_return = calculate_portfolio_return(self.portfolio_return, self.weights, expected_asset_returns)

        # Calculate Sharpe Ratio
        # self.portfolio_sharpe_ratio = ((self.portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5))

        self.weights = pd.DataFrame(self.weights)
        self.weights.index = asset_names
        self.weights = self.weights.T
