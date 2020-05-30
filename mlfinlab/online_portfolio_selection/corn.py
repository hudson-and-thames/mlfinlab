# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.base import OLPS


class CORN(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29. <https://dl.acm.org/doi/abs/10.1145/1961189.1961193>`_

    Correlation Driven Nonparametric Learning finds similar windows from the past and looks to
    create portfolio weights that will maximize returns in the similar sets.
    """

    def __init__(self, window, rho):
        """
        Initializes Correlation Driven Nonparametric Learning with the given window and rho value.

        :param window: (int) Number of windows to look back for similarity sets. Windows can be set
                             to any values but typically work well in a shorter term of [1, 7].
        :param rho: (float) Threshold for similarity with range of [-1, 1].
                            Lower rho values will classify more periods as being similar, and higher
                            values will be more strict on identifying a period as similarly correlated.
                            Rho values between [0, 0.2] typically had higher results.
        """
        self.window = window
        self.rho = rho
        self.corr_coef = None  # (np.array) Rolling correlation coefficients.
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """
        super(CORN, self)._initialize(asset_prices, weights,
                                      resample_by)
        # Check that window value is an integer.
        if not isinstance(self.window, int):
            raise ValueError("Window value must be an integer.")

        # Check that window value is at least 1.
        if self.window < 1:
            raise ValueError("Window value must be greater than or equal to 1.")

        # Check that rho is between -1 and 1.
        if self.rho < -1 or self.rho > 1:
            raise ValueError("Rho must be between -1 and 1.")
        self.corr_coef = self._calc_rolling_corr_coef()

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """
        # Create similar set.
        similar_set = []

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

            if similar_set:
                # Choose the corresponding relative return periods.
                optimize_array = self.relative_return[similar_set]
                new_weights = self._fast_optimize(optimize_array)
        return new_weights

    def _fast_optimize(self, optimize_array):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :return: (np.array) Weights that maximize the returns for the given array.
        """
        # Initialize guess.
        weights = self._uniform_weight()

        # Use np.log and np.sum to make the cost function a convex function.
        # Multiplying continuous returns equates to summing over the log returns.
        def _objective(weight):
            return -np.sum(np.log(np.dot(optimize_array, weight)))

        # Derivative of the objective function.
        def _derivative(weight):
            total_returns = np.dot(optimize_array, weight)
            return -np.dot(1 / total_returns, optimize_array)

        # Weight bounds.
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # Sum of weights is 1.
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(_objective, weights, method='SLSQP', bounds=bounds,
                               constraints=const, jac=_derivative)
        return problem.x

    def _calc_rolling_corr_coef(self):
        """
        Calculates the rolling correlation coefficient for a given relative return and window

        :return: (np.array) Rolling correlation coefficient over a given window.
        """
        # Flatten the array.
        flattened = self.relative_return.flatten()

        # Set index of rolled window.
        idx = np.arange(self.number_of_assets * self.window)[None, :] + self.number_of_assets * \
              np.arange(self.length_of_time - self.window + 1)[:, None]

        # Retrieve the results of the rolled window.
        rolled_returns = flattened[idx]

        # Calculate correlation coefficient.
        with np.errstate(divide='ignore', invalid='ignore'):
            rolling_corr_coef = np.nan_to_num(np.corrcoef(rolled_returns), nan=0)
        return rolling_corr_coef
