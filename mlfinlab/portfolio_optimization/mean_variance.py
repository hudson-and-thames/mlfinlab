'''
This module implements the classic mean-variance optimisation techniques for calculating the efficient frontier.
It uses typical quadratic optimisers to generate optimal portfolios for different objective functions.
'''

import numpy as np
import pandas as pd


class MeanVarianceOptimisation:
    '''
    This class contains a variety of methods dealing with different solutions to the mean variance optimisation
    problem.
    '''

    def __init__(self):
        self.weights = list()

    def allocate(self, asset_prices, solution='inverse_variance', resample_by='B'):
        '''
        Calculate the portfolio asset allocations using the method specified.

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param solution: (str) the type of solution/algorithm to use to calculate the weights
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        '''

        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe")
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

        # Calculate returns
        asset_returns = self._calculate_returns(asset_prices, resample_by=resample_by)
        assets = asset_prices.columns

        if solution == 'inverse_variance':
            cov = asset_returns.cov()
            self.weights = self._inverse_variance(covariance=cov)
        else:
            raise ValueError("Unknown solution string specified. Supported solutions - inverse_variance.")
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = assets
        self.weights = self.weights.T

    @staticmethod
    def _calculate_returns(asset_prices, resample_by):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :return: (pd.Dataframe) stock returns
        '''

        asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns

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
