import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class BAH:
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    The Buy and Hold strategy one invests wealth among the market with an initial portfolio of weights and holds
    the portfolio till the end. The manager only buys the assets at the beginning of the first period and does
    not rebalance in subsequent periods.
    """

    def __init__(self, calculate_expected_returns='mean'):
        """
        Constructor.
        :param calculate_expected_returns: (str) the method to use for calculation of expected returns.
        Currently supports "mean" and "exponential"
        """

        self.weights = list()
        self.portfolio_risk = None
        self.portfolio_return = None
        self.portfolio_sharpe_ratio = None
        self.calculate_expected_returns = calculate_expected_returns
        self.returns_estimator = ReturnsEstimation()

    def allocate(self,
                 asset_names,
                 asset_prices=None,
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

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        number_of_assets = len(asset_names)

        if weights is None:
            self.weights = np.linspace(0, 1, num=number_of_assets)
        else:
            self.weights = weights

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
            covariance_matrix = returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        # Calculate the expected returns if the user does not supply any returns
        if expected_asset_returns is None:
            if self.calculate_expected_returns == "mean":
                expected_asset_returns = self.returns_estimator.calculate_mean_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = self.returns_estimator.calculate_exponential_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")
        expected_asset_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))

        # Calculate the portfolio risk and return if it has not been calculated
        if self.portfolio_risk is None:
            self.portfolio_risk = np.dot(self.weights, np.dot(cov.values, self.weights.T))
        if self.portfolio_return is None:
            self.portfolio_return = np.dot(self.weights, expected_asset_returns)
        self.portfolio_sharpe_ratio = ((self.portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5))

        self.weights = pd.DataFrame(self.weights)
        self.weights.index = asset_names
        self.weights = self.weights.T
