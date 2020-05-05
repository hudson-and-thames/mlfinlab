# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class CorrelationDrivenNonparametricLearning(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29.<https://dl.acm.org/doi/abs/10.1145/1961189.1961193>`_

    Correlation Driven Nonparametric Learning finds similar windows from the past and looks to
    create portfolio weights that will maximize returns in the similar sets.
    """
    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, window, rho):
        """
        Initializes Correlation Driven Nonparametric Learning with the given window and rho value.

        :param window: (int) Number of windows to look back for similarity sets.
        :param rho: (float) Threshold for similarity.
        """
        self.window = window
        self.rho = rho
        self.corr_coef = None
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices.
        """
        super(CorrelationDrivenNonparametricLearning, self)._initialize(asset_prices, weights,
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
        self.corr_coef = self.calculate_rolling_correlation_coefficient()

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weights: (np.array) Predicted weights.
        """
        similar_set = []
        new_weights = self._uniform_weight()
        if time - 1 > self.window:
            activation_fn = np.zeros(self.length_of_time)
            for i in range(self.window + 1, time - 1):
                if self.corr_coef[i - 1][time - 1] > self.rho:
                    similar_set.append(i)
            if similar_set:
                # put 1 for the values in the set
                activation_fn[similar_set] = 1
                new_weights = self._optimize(relative_return, activation_fn, cp.SCS)
        return new_weights

    def _fast_optimize(self, optimize_array):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :return: problem.x: (np.array) Weights that maximize the returns for the given array.
        """
        # Initialize guess.
        weights = self._uniform_weight()

        # Use np.log and np.sum to make the cost function a convex function.
        # Multiplying continuous returns equates to summing over the log returns.
        def _objective(weight):
            return -np.sum(np.log(np.dot(optimize_array, weight)))

        # Weight bounds.
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # Sum of weights is 1.
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(_objective, weights, method='SLSQP', bounds=bounds, constraints=const)
        return problem.x

    def calculate_rolling_correlation_coefficient(self, _relative_return):
        """
        Calculates the rolling correlation coefficient for a given relative return and window

        :param _relative_return: (np.array) relative returns of a certain time period specified by the strategy
        :return rolling_corr_coef: (np.array) rolling correlation coefficient over a given window
        """
        # take the log of the relative return
        # first calculate the rolling window the relative return
        # sum the data which returns the log of the window's return
        # take the exp to revert back to the original window's returns
        # calculate the correlation coefficient for the different window's overall returns
        rolling_corr_coef = np.corrcoef(np.exp(np.log(pd.DataFrame(_relative_return)).rolling(self.window).sum()))
        return rolling_corr_coef
