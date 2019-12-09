'''
This module implements some classic mean-variance optimisation techniques for calculating the efficient frontier.
It uses typical quadratic optimisers to generate optimal portfolios for different objective functions.
'''

import numpy as np
import pandas as pd
import cvxpy as cp
from mlfinlab.portfolio_optimization.returns_estimation import calculate_returns, calculate_mean_historical_returns, calculate_exponential_historical_returns


class MeanVarianceOptimisation:
    '''
    This class contains a variety of methods dealing with different solutions to the mean variance optimisation
    problem.
    '''

    def __init__(self, calculate_expected_returns='mean'):
        self.weights = list()
        self.portfolio_risk = None
        self.portfolio_return = None
        self.calculate_expected_returns=calculate_expected_returns

    def allocate(self,
                 asset_names,
                 asset_prices=None,
                 expected_asset_returns=None,
                 covariance_matrix=None,
                 solution='inverse_variance',
                 risk_free_rate=0.05,
                 resample_by=None):
        # pylint: disable=invalid-name, too-many-branches
        '''
        Calculate the portfolio asset allocations using the method specified.

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param expected_asset_returns: (list) a list of mean stock returns (mu)
        :param covariance_matrix: (pd.Dataframe/numpy matrix) user supplied covariance matrix of asset returns (sigma)
        :param solution: (str) the type of solution/algorithm to use to calculate the weights
        :param risk_free_rate: (float) the rate of return for a risk-free asset.
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        '''

        if asset_prices is None and expected_asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or expected returns "
                             "and a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        # Calculate the expected returns if the user does not supply any returns
        if expected_asset_returns is None and solution not in {'inverse_variance', 'min_volatility'}:
            if self.calculate_expected_returns == "mean":
                expected_asset_returns = calculate_mean_historical_returns(asset_prices=asset_prices,
                                                                          resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = calculate_exponential_historical_returns(asset_prices=asset_prices,
                                                                                 resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")
            expected_asset_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))
            if (expected_asset_returns == np.ones(expected_asset_returns.shape) * expected_asset_returns.mean()).all():
                expected_asset_returns[-1, 0] += 1e-5

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            returns = calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
            covariance_matrix = returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        if solution == 'inverse_variance':
            self.weights = self._inverse_variance(covariance=cov)
        elif solution == 'min_volatility':
            self.weights, self.portfolio_risk = self._min_volatility(covariance=cov, num_assets=len(asset_names))
        elif solution == 'max_sharpe':
            self.weights, self.portfolio_risk = self._max_sharpe(covariance=cov,
                                                                 expected_returns=expected_asset_returns,
                                                                 risk_free_rate=risk_free_rate,
                                                                 num_assets=len(asset_names))
        else:
            raise ValueError("Unknown solution string specified. Supported solutions - inverse_variance.")
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = asset_names
        self.weights = self.weights.T

    @staticmethod
    def _inverse_variance(covariance):
        '''
        Calculate weights using inverse-variance allocation

        :param covariance: (pd.Dataframe) covariance dataframe of asset returns
        :return: (np.array) array of portfolio weights
        '''

        ivp = 1. / np.diag(covariance)
        ivp /= ivp.sum()
        return ivp

    @staticmethod
    def _min_volatility(covariance, num_assets):
        '''

        :param covariance:
        :param num_assets:
        :return:
        '''

        weights = cp.Variable(num_assets)
        risk = cp.quad_form(weights, covariance)

        # Define optimisation objective and constraints
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve()
        return weights.value, risk.value

    def _max_sharpe(self, covariance, expected_returns, risk_free_rate, num_assets):
        '''

        :param covariance:
        :param expected_returns:
        :param risk_free_rate:
        :param num_assets:
        :return:
        '''
        
        y = cp.Variable(num_assets)
        tau = cp.Variable(1)
        risk = cp.quad_form(y, covariance)

        # Define optimisation objective and constraints
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum((expected_returns - risk_free_rate).T @ y) == 1,
            cp.sum(y) == tau,
            y >= 0,
            tau >= 0
        ]

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve()
        weights = y.value / tau.value
        return weights, risk.value

    @staticmethod
    def _efficient_risk():
        return

    @staticmethod
    def _efficient_return():
        return

    @staticmethod
    def plot_efficient_frontier():
        return
