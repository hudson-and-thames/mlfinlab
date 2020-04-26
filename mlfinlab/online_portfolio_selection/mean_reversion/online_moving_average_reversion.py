# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class OnlineMovingAverageReversion(OLPS):
    """
    This class implements the Online Moving Average Reversion strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    Online Moving Average Reversion reverts to the SMA or EMA of the underlying assets based on the given threshold.
    """

    def __init__(self, reversion_method, epsilon, window=None, alpha=None):
        self.epsilon = epsilon  # (float) reversion threshold >= 1
        self.window = window  # (int) number of windows to calculate Simple Moving Average
        self.alpha = alpha  # (float) ratio between 0 and 1 for Exponentially Weighted Average
        self.reversion_method = reversion_method  # (int) 1 for SMA, 2 for EWA
        self.moving_average_reversion = None  # (np.array) calculated moving average reversion to speed up online learning
        super().__init__()

    # intialize moving average reversion
    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        super(OnlineMovingAverageReversion, self)._initialize(asset_prices, weights, resample_by)

        # pre-calculate moving_average_reversion to speed up
        self.moving_average_reversion = self.calculate_rolling_moving_average(self.asset_prices, self.window,
                                                                              self.reversion_method, self.alpha)

    def _update_weight(self, time):
        """
        Updates portfolio weights

        :param time: (int) current time period
        :return (None) sets new weights to be the same as old weights
        """
        # return predetermined weights for time periods with no significant data
        if self.reversion_method == 1 and time < self.window or time == 0:
            return self.weights
        # get predicted change through SMA or EWA
        predicted_change = self.moving_average_reversion[time]
        # calculate the mean of the predicted change
        mean_relative = np.mean(predicted_change)
        # portfoliio weights of mean prediction
        mean_change = np.ones(self.number_of_assets) * mean_relative
        # loss function to switch mean reversion strategy
        loss_fn = max(0, (self.epsilon - np.dot(self.weights, predicted_change)))
        # if loss function is 0, set multiplicative constant to zero
        if loss_fn == 0:
            lambd = 0
        # if not, adjust lambda, a multiplicative constant
        else:
            lambd = loss_fn / (np.linalg.norm(predicted_change - mean_change) ** 2)
        new_weights = self.weights + lambd * (predicted_change - mean_change)
        # project to simplex as most likely weights will not sum to 1
        return self._simplex_projection(new_weights)

    @staticmethod
    def calculate_rolling_moving_average(_asset_prices,
                                         _window,
                                         _reversion_method,
                                         _alpha):
        """
        Calculates the rolling moving average for Online Moving Average Reversion

        :param _asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param _window: (int) number of market windows
        :param _reversion_method: (int) number that represents the reversion method
                                        1: SMA, 2: EMA
        :param _alpha: (int) exponential weight for the second reversion method
        :return rolling_ma: (np.array) rolling moving average for the given reversion method
        """
        # MAR-1 reversion method: Simple Moving Average
        if _reversion_method == 1:
            rolling_ma = np.array(_asset_prices.rolling(_window).apply(lambda x: np.sum(x) / x[0] / _window))
        # MAR-2 reversion method: Exponential Moving Average
        elif _reversion_method == 2:
            rolling_ma = np.array(_asset_prices.ewm(alpha=_alpha, adjust=False).mean() / _asset_prices)
        return rolling_ma
