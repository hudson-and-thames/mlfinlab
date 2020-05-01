# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class OnlineMovingAverageReversion(OLPS):
    """
    This class implements the Online Moving Average Reversion strategy. It is reproduced with
    modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December YEAR), 33 pages. <https://arxiv.org/abs/1212.2129>.

    Online Moving Average Reversion reverts to the SMA or EMA of the underlying assets based on
    the given threshold.
    """

    def __init__(self, reversion_method, epsilon, window=None, alpha=None):
        """
        Initializes Online Moving Average Reversion with the given reversion method, epsilon,
        window, and alpha.

        :param reversion_method: (int) 1 for SMA, 2 for EWA.
        :param epsilon: (float) reversion threshold >= 1.
        :param window: (int) number of windows to calculate Simple Moving Average.
        :param alpha: (float) ratio between 0 and 1 for Exponentially Weighted Average.
        """
        self.epsilon = epsilon
        self.window = window
        self.alpha = alpha
        self.reversion_method = reversion_method
        self.moving_average_reversion = None  # Calculated moving average.
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices.
        """
        super(OnlineMovingAverageReversion, self)._initialize(asset_prices, weights, resample_by)

        # Check that epsilon values are correct.
        if self.epsilon < 1:
            raise ValueError("Epsilon values must be greater or equal to 1.")

        # Check that window is at least 1.
        if self.reversion_method == 1 and self.window < 1:
            raise ValueError("Window must be at least 1.")

        if self.reversion_method == 2 and (self.alpha < 0 or self.alpha > 1):
            raise ValueError("Alpha must be between 0 and 1.")

        # Calculate moving average reversion.
        self.moving_average_reversion = self._calculate_rolling_moving_average(self.asset_prices,
                                                                               self.window,
                                                                               self.reversion_method,
                                                                               self.alpha)

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weights: (np.array) Predicted weights.
        """
        # Return predetermined weights for time periods with no significant data.
        if self.reversion_method == 1 and time < self.window or time == 0:
            return self.weights
        # Get predicted change through SMA or EWA.
        predicted_change = self.moving_average_reversion[time]
        # Calculate the mean of the predicted change.
        mean_relative = np.mean(predicted_change)
        # Portfolio weights of mean prediction.
        mean_change = np.ones(self.number_of_assets) * mean_relative
        # Loss function to switch mean reversion strategy.
        loss_fn = max(0, (self.epsilon - np.dot(self.weights, predicted_change)))
        # If loss function is 0, set multiplicative constant to zero.
        if loss_fn == 0:
            lambd = 0
        # If not, adjust lambda, a multiplicative constant.
        else:
            lambd = loss_fn / (np.linalg.norm(predicted_change - mean_change) ** 2)
        new_weights = self.weights + lambd * (predicted_change - mean_change)
        # Projects new weights to simplex domain.
        new_weights = self._simplex_projection(new_weights)
        return new_weights

    @staticmethod
    def _calculate_rolling_moving_average(asset_prices, window, reversion_method, alpha):
        """
        Calculates the rolling moving average for Online Moving Average Reversion.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param window: (int) Number of market windows.
        :param reversion_method: (int) Reversion method. 1 : SMA, 2: EMA.
        :param alpha: (int) Exponential weight for the second reversion method.
        :return rolling_ma: (np.array) Rolling moving average for the given reversion method.
        """
        # MAR-1 reversion method: Simple Moving Average.
        if reversion_method == 1:
            rolling_ma = np.array(asset_prices.rolling(window).apply(lambda x: np.sum(x) / x[0] / window))
        # MAR-2 reversion method: Exponential Moving Average.
        elif reversion_method == 2:
            rolling_ma = np.array(asset_prices.ewm(alpha=alpha, adjust=False).mean() / asset_prices)
        return rolling_ma
