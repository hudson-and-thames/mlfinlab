# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class OLMAR(OLPS):
    """
    This class implements the Online Moving Average Reversion strategy. It is reproduced with
    modification from the following paper:
    `Li, Bin & Hoi, Steven. (2012). On-Line Portfolio
    Selection with Moving Average Reversion. Proceedings of the 29th International Conference on
    Machine Learning, ICML 2012. 1. <https://arxiv.org/pdf/1206.4626.pdf>`_

    Online Moving Average Reversion reverts to the SMA or EMA of the underlying assets based on
    the given threshold.
    """

    def __init__(self, reversion_method, epsilon, window=None, alpha=None):
        """
        Initializes Online Moving Average Reversion with the given reversion method, epsilon,
        window, and alpha.

        :param reversion_method: (int) 1 for SMA, 2 for EWA. Typically 2 has higher returns than 1,
                                       but this is dependant on the dataset.
        :param epsilon: (float) Reversion threshold with range [1, inf). OLMAR methods typically do
                                not rely on epsilon values as it is more dependant on the window and
                                alpha value; however, epsilon of 20 seems to be a safe choice to
                                avoid extreme values.
        :param window: (int) Number of windows to calculate Simple Moving Average with range [1, inf).
                             This parameter depends on the data. If the data has a shorter mean reversion
                             trends, then window of 6 can have high returns. If the data has a longer
                             mean reversion trend, then values close to 21 can have high returns.
        :param alpha: (float) Exponential ratio for Exponentially Weighted Average with range (0, 1).
                              Larger alpha value indicates more importance placed on recent prices,
                              and smaller alpha value indicates more importance placed on the past.
        """
        self.epsilon = epsilon
        self.window = window
        self.alpha = alpha
        self.reversion_method = reversion_method
        self.moving_average_reversion = None  # (np.array) Calculated moving average.
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """
        super(OLMAR, self)._initialize(asset_prices, weights, resample_by)

        # Check that reversion method is either 1 or 2.
        if self.reversion_method not in [1, 2]:
            raise ValueError("Reversion method must be either 1 or 2.")

        # Check that epsilon values are correct.
        if self.epsilon < 1:
            raise ValueError("Epsilon values must be greater or equal to 1.")

        # Check that window is at least 1.
        if self.reversion_method == 1 and self.window < 1:
            raise ValueError("Window must be at least 1.")

        # Chat that alpha is between 0 and 1 for method 2.
        if self.reversion_method == 2 and (self.alpha < 0 or self.alpha > 1):
            raise ValueError("Alpha must be between 0 and 1.")

        # Calculate moving average reversion.
        self.moving_average_reversion = self._calculate_rolling_moving_average()

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
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
            check = np.linalg.norm(predicted_change - mean_change) ** 2
            if check != 0:
                lambd = loss_fn / check
            else:
                lambd = 0
        new_weights = self.weights + lambd * (predicted_change - mean_change)

        # Projects new weights to simplex domain.
        new_weights = self._simplex_projection(new_weights)
        return new_weights

    def _calculate_rolling_moving_average(self):
        """
        Calculates the rolling moving average for Online Moving Average Reversion.

        :return: (np.array) Rolling moving average for the given reversion method.
        """
        # MAR-1 reversion method: Simple Moving Average.
        if self.reversion_method == 1:
            rolling_ma = np.array(self.asset_prices.rolling(self.window).apply(
                lambda x: np.sum(x) / x[0] / self.window, raw=True))
        # MAR-2 reversion method: Exponential Moving Average.
        else:
            rolling_ma = np.array(
                self.asset_prices.ewm(alpha=self.alpha, adjust=False).mean() / self.asset_prices)
        return rolling_ma
