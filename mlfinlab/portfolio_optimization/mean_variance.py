'''
This module implements the classic mean-variance optimisation techniques for calculating the efficient frontier.
It uses typical quadratic optimisers to generate optimal portfolios for different objective functions.
'''

import numpy as np
import pandas as pd


class MeanVarianceOptimisation:

    def __init__(self):
        return

    def allocate(self, asset_prices, solution='inverse_variance'):
        '''

        :param asset_prices: (pd.Dataframe/np.array) the matrix of historical asset prices (daily close)
        :param solution: (str) the type of solution/algorithm to use to calculate the weights
        '''

        if not isinstance(asset_prices, pd.DataFrame):
            asset_prices = pd.DataFrame(asset_prices)

        self.weights = []
        assets = asset_prices.columns
        if solution == 'inverse_variance':
            self.weights = self._inverse_variance(asset_prices=asset_prices)
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = assets
        self.weights = self.weights.T

    def _inverse_variance(self, asset_prices):
        '''

        :param asset_prices: (pd.Dataframe/np.array) the matrix of historical asset prices (daily close)
        :return: (np.array) array of portfolio weights
        '''

        cov = asset_prices.cov()
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _min_volatility(self):
        return

    def _max_sharpe(self):
        return