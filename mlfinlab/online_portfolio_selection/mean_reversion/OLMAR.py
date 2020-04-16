# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class OLMAR(OLPS):
    """
    This class implements the Online Moving Average Reversion Strategy.
    """

    def __init__(self, epsilon=2, window=2, alpha=.9, reversion_method=1):
        """
        Constructor.
        """
        # check that epsilon is > 1
        # check that window is >= 1
        # check that alpha is (0,1)
        # check that reversion_method is either 1 or 2
        # if optimization_method == 2 then reversion_method doesn't matter
        self.epsilon = epsilon
        self.window = window
        self.alpha = alpha
        self.reversion_method = reversion_method
        self.moving_average_reversion = None
        super().__init__()

    # intialize moving average reversion
    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        """

        :param _asset_prices:
        :param _weights:
        :param _portfolio_start:
        :param _resample_by:
        :return:
        """
        self.moving_average_reversion = self.calculate_rolling_moving_average(_asset_prices, self.window,
                                                                              self.reversion_method, self.alpha)
        super(OLMAR, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        if self.reversion_method == 1 and _time < self.window:
            return self.weights
        # calculate price relative
        predicted_change = self.moving_average_reversion[_time]
        # different OLMAR methods
        mean_relative = np.mean(predicted_change)
        mean_change = np.ones(self.number_of_assets) * mean_relative
        lambd = max(0, (self.epsilon - np.dot(_weights, predicted_change)) / (
            np.linalg.norm(predicted_change - mean_change) ** 2))

        new_weights = _weights + lambd * (predicted_change - mean_change)
        if np.isnan(new_weights).any():
            raise ValueError()
        # if not in simplex domain
        return self.simplex_projection(new_weights)
