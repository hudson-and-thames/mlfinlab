import numpy as np
import pandas as pd


class MeanVarianceOptimisation:

    def __init__(self):
        return

    def allocate(self, asset_prices, solution='inverse_variance'):
        '''

        :param asset_prices:
        :param solution:
        :return:
        '''

        if not isinstance(asset_prices, pd.DataFrame):
            asset_prices = pd.DataFrame(asset_prices)

        self.weights = []
        if solution == 'inverse_variance':
            self.weights = self._inverse_variance(asset_prices=asset_prices)

    def _inverse_variance(self, asset_prices):
        '''

        :param asset_prices:
        :return:
        '''

        cov = asset_prices.cov()
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _min_volatility(self):
        return

    def _max_sharpe(self):
        return