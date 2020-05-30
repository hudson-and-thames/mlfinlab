# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.corn import CORN


class SCORN(CORN):
    """
    This class implements the Symmetric Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Symmetric Correlation Driven Nonparametric Learning is an extension of the CORN strategy
    proposed by Yang Wang and Dong Wang. SCORN looks to not only maximize the returns for the similar
    periods but also minimize the losses from the negatively correlated periods.
    """

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """
        # Create similar set.
        similar_set = []

        # Creat opposite set.
        opposite_set = []

        # Default is uniform weights.
        new_weights = self._uniform_weight()

        # Calculate for similar sets if time is greater or equal to window size.
        if time >= self.window:
            # Iterate through past windows.
            for past_time in range(time - self.window + 1):
                # Check for windows that are above rho.

                if self.corr_coef[time - self.window + 1][past_time] > self.rho:
                    # Append the time for similar set.
                    similar_set.append(past_time + self.window)

                # Check for windows that are below rho.
                elif self.corr_coef[time - self.window + 1][past_time] < -self.rho:
                    # Append the time for opposite set.
                    opposite_set.append(past_time + self.window)

            # If either set exists, continue to return new weights.
            if similar_set or opposite_set:
                # Choose the corresponding relative return periods.
                similar = self.relative_return[similar_set]
                opposite = self.relative_return[opposite_set]
                new_weights = self._scorn_optimize(similar, opposite)
        return new_weights

    def _scorn_optimize(self, similar, opposite):
        """
        Calculates weights that maximize returns over the given array.

        :param similar: (np.array) Relative returns of similar periods.
        :param opposite: (np.array) Relative returns of inversely similar periods.
        :return: (np.array) Weights that maximize the returns for the given array.
        """
        # Initialize guess.
        weights = self._uniform_weight()

        # Use np.log and np.sum to make the cost function a convex function.
        # Multiplying continuous returns equates to summing over the log returns.
        def _objective(weight):
            return -np.sum(np.log(np.dot(similar, weight))) + np.sum(np.log(np.dot(opposite, weight)))

        # Derivative of the objective function.
        def _derivative(weight):
            similar_returns = np.dot(similar, weight)
            opposite_returns = np.dot(opposite, weight)
            return -np.dot(1 / similar_returns, similar) + np.dot(1 / opposite_returns, opposite)

        # Weight bounds.
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # Sum of weights is 1.
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(_objective, weights, method='SLSQP', bounds=bounds, constraints=const, jac=_derivative)
        return problem.x
