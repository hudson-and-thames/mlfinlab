# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class RMR(OLPS):
    """
    This class implements the Confidence Weighted Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    `D. Huang, J. Zhou, B. Li, S. C. H. Hoi and S. Zhou, "Robust Median Reversion Strategy for
    Online Portfolio Selection," in IEEE Transactions on Knowledge and Data Engineering, vol. 28,
    no. 9, pp. 2480-2493, 1 Sept. 2016. <https://www.ijcai.org/Proceedings/13/Papers/296.pdf>`_

    Robust Median Reversion uses a L1-median of historical prices to predict the next time's
    price relative returns. The new weights is then regularized to retain previous portfolio
    information but also seeks to maximize returns from the predicted window.
    """

    def __init__(self, epsilon, n_iteration, window, tau=0.001):
        """
        Initializes Robust Median Reversion with the given epsilon, n_iteration, window, and tau values.

        :param epsilon: (float) Reversion threshold with range [1, inf). Values of [15, 25] had the
                                highest returns for the original dataset provided by the authors.
        :param n_iteration: (int) Maximum number of iterations with range [2, inf). Iteration of 200
                                  produced a adequate balance between performance and computational
                                  time for the strategy.
        :param window: (int) Size of window with range [2, inf). Typically window of 2, 7, or 21
                             produced the highest results. This parameter depends on the underlying
                             data's price movements and mean reversion tendencies.
        :param tau: (float) Toleration level with range [0, 1). It is suggested to keep the toleration
                            level at 0.001 for computational efficiency.
        """
        self.epsilon = epsilon
        self.n_iteration = n_iteration
        self.window = window
        self.tau = tau
        self.np_asset_prices = None  # (np.array) Asset prices converted to np.array.
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """
        super(RMR, self)._initialize(asset_prices, weights, resample_by)

        # Check that epsilon value is correct.
        if self.epsilon < 1:
            raise ValueError("Epsilon values must be greater than 1.")

        # Check that the n_iteration is an integer.
        if not isinstance(self.n_iteration, int):
            raise ValueError("Number of iterations must be an integer.")

        # Check that the number of iterations is greater or equal to 2.
        if self.n_iteration < 2:
            raise ValueError("Number of iterations must be greater or equal to 2.")

        # Check that the window value is an integer.
        if not isinstance(self.window, int):
            raise ValueError("Window must be an integer.")

        # Check that the window value is greater or equal to 2.
        if self.window < 2:
            raise ValueError("Window must be greater or equal to 2.")

        # Check that tau is within range of [0, 1).
        if self.tau < 0 or self.tau >= 1:
            raise ValueError("Tau must be greater or equal to 0 and less than 1.")

        self.np_asset_prices = np.array(self.asset_prices)

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """
        # Until the relative time window, return original weights.
        if time < self.window - 1:
            return self.weights

        # Set the current predicted relatives value.
        current_prediction = self._calculate_predicted_relatives(time)

        # Set the deviation from the mean of current prediction.
        predicted_deviation = current_prediction - np.ones(self.number_of_assets) * np.mean(
            current_prediction)

        # Calculate alpha, the lagrangian multiplier.
        norm2 = np.linalg.norm(predicted_deviation, ord=1) ** 2

        # If norm2 is zero, return previous weights.
        if norm2 == 0:
            return self.weights
        alpha = np.minimum(0, (current_prediction * self.weights - self.epsilon) / norm2)

        # Update new weights.
        new_weights = self.weights - alpha * predicted_deviation

        # Project to simplex domain.
        new_weights = self._simplex_projection(new_weights)

        return new_weights

    def _calculate_predicted_relatives(self, time):
        # pylint: disable=unsubscriptable-object
        """
        Calculates the predicted relatives using l1 median.

        :param time: (int) Current time.
        :return: (np.array) Predicted relatives using l1 median.
        """
        # Calculate the L1 median of the price window.
        price_window = self.np_asset_prices[time - self.window + 1:time + 1]
        curr_prediction = np.median(price_window, axis=0)

        # Iterate until the maximum iteration allowed.
        for _ in range(self.n_iteration - 1):
            prev_prediction = curr_prediction
            # Transform mu according the Modified Weiszfeld Algorithm
            curr_prediction = self._transform(curr_prediction, price_window)

            # If null value or condition is satisfied, break.
            if curr_prediction.size == 0 or np.linalg.norm(prev_prediction - curr_prediction, ord=1) \
                    <= self.tau * np.linalg.norm(curr_prediction, ord=1):
                curr_prediction = prev_prediction
                break

        # Divide by the current time's price.
        predicted_relatives = curr_prediction / price_window[-1]

        return predicted_relatives

    @staticmethod
    def _transform(old_mu, price_window):
        """
        Calculates L1 median approximation by using the Modified Weiszfeld Algorithm.

        :param old_mu: (np.array) Current value of the predicted median value.
        :param price_window: (np.array) A window of prices provided by the user.
        :return: (np.array) New updated l1 median approximation.
        """
        # Calculate the difference set.
        diff = price_window - old_mu

        # Remove rows with all zeros.
        non_mu = diff[~np.all(diff == 0, axis=1)]

        # Edge case for identical price windows.
        if non_mu.shape[0] == 0:
            return non_mu

        # Number of zeros.
        n_zero = diff.shape[0] - non_mu.shape[0]

        # Calculate eta.
        eta = 0 if n_zero == 0 else 1

        # Calculate l1 norm of non_mu.
        l1_norm = np.linalg.norm(non_mu, ord=1, axis=1)

        # Calculate tilde.
        tilde = 1 / np.sum(1 / l1_norm) * np.sum(np.divide(non_mu.T, l1_norm), axis=1)

        # Calculate gamma.
        gamma = np.linalg.norm(
            np.sum(np.apply_along_axis(lambda x: x / np.linalg.norm(x, ord=1), 1, non_mu), axis=0),
            ord=1)

        # Calculate next_mu value.
        with np.errstate(invalid='ignore'):
            next_mu = np.maximum(0, 1 - eta / gamma) * tilde + np.minimum(1, eta / gamma) * old_mu
        return tilde if eta == 0 else next_mu
