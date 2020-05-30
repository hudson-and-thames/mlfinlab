# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.corn import CORN


class FCORN(CORN):
    """
    This class implements the Functional Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Functional Correlation Driven Nonparametric Learning is an extension of the SCORN strategy.
    Unlike SCORN that sets a certain threshold for the objective function, FCORN places an
    arbitrary weight to all relative returns period to emphasize certain weights. If the given
    lambda value approaches infinity, then FCORN returns the same values as the SCORN strategy.
    """

    def __init__(self, window, rho, lambd):
        """
        Initializes Functional Correlation Driven Nonparametric Learning with the given window,
        rho, and lambd value.

        :param window: (int) Number of windows to look back for similarity sets. Windows can be set
                             to any values but typically work well in a shorter term of [1, 7].
        :param rho: (float) Threshold for similarity. Rho should set in the range of [-1, 1].
                            Lower rho values will classify more periods as being similar, and higher
                            values will be more strict on identifying a period as similarly correlated.
                            Rho values between [0.4, 0.8] typically had higher results.
        :param lambd: (float) Scale factor for the sigmoid activation function. Lambd can be any
                              value. If lambd approaches infinity, FCORN will have the same results
                              as SCORN, and if lambd approaches negative infinity, FCORN produces
                              similar results as a Constant Rebalanced Portfolio. From experimental
                              results, lambd of 1 was optimal for some datasets.
        """
        self.lambd = lambd
        super().__init__(window=window, rho=rho)

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """
        # Default is uniform weights.
        new_weights = self._uniform_weight()

        # Calculate for similar sets if time is greater or equal to window size.
        if time >= self.window:
            # Create activation_fn to scale the relative returns during the optimization process.
            activation_fn = np.zeros(time + 1)

            # Iterate through past windows.
            for past_time in range(time - self.window + 1):
                # Get correlation between the two windows.
                corr = self.corr_coef[time - self.window + 1][past_time]

                # If correlation is non-negative.
                if corr >= 0:
                    # Set corresponding activation function
                    activation_fn[past_time + self.window] = self._sigmoid(
                        self.lambd * (corr - self.rho))

                # If correlation is negative.
                else:
                    activation_fn[past_time + self.window] = self._sigmoid(
                        self.lambd * (corr + self.rho)) - 1

            # Negative activation function to minimize final objective function.
            activation_fn = -activation_fn

            # Current relative returns.
            curr_time = np.asfortranarray(self.relative_return[:time + 1])

            # Calculate new weights based on activation_fn and relatives returns to date.
            new_weights = self._fcorn_optimize(activation_fn, curr_time)
        return new_weights

    def _fcorn_optimize(self, activation_fn, relative_return):
        """
        Calculates weights that maximize returns over the given array.

        :param activation_fn: (np.array) Scaling factor for optimization equation.
        :param relative_return: (np.array) Array of relative returns until the current period
        :return: (np.array) Weights that maximize the returns for the given array.
        """
        # Initialize guess.
        weights = self._uniform_weight()

        # Use np.log and np.sum to make the cost function a convex function.
        # Multiplying continuous returns equates to summing over the log returns.
        def _objective(weight):
            return np.dot(activation_fn, np.log(np.dot(relative_return, weight)))

        def _derivative(weight):
            total_returns = np.dot(relative_return, weight)
            return np.dot(activation_fn / total_returns, relative_return)

        # Weight bounds.
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # Sum of weights is 1.
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        problem = opt.minimize(_objective, weights, method='SLSQP', bounds=bounds,
                               constraints=const, jac=_derivative)
        return problem.x

    @staticmethod
    def _sigmoid(val):
        """
        Generates the resulting sigmoid function

        :param val: (float) Input for
        :return: (float) sigmoid(x)
        """
        # Calculate sigmoid.
        res = 1 / (1 + np.exp(-val))
        return res
