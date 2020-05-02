# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class CorrelationDrivenNonparametricLearning(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    'Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29.<https://dl.acm.org/doi/abs/10.1145/1961189.1961193>_'

    Correlation Driven Nonparametric Learning finds similar windows from the past and looks to
    create a portfolio weights that will maximize returns in the similar sets.
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
        super(CorrelationDrivenNonparametricLearning, self)._initialize(asset_prices, weights, resample_by)
        self.corr_coef = self.calculate_rolling_correlation_coefficient()

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weights: (np.array) Predicted weights.
        """
        similar_set = []
        new_weights = self._uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            activation_fn = np.zeros(self.final_number_of_time)
            for i in range(self.window + 1, _time - 1):
                if self.corr_coef[i - 1][_time - 1] > self.rho:
                    similar_set.append(i)
            if similar_set:
                # put 1 for the values in the set
                activation_fn[similar_set] = 1
                new_weights = self._optimize(_relative_return, activation_fn, cp.SCS)
        return new_weights

    # optimize the weight that maximizes the returns
    def _optimize(self, _optimize_array, _activation_fn, _solver=None):
        length_of_time = _optimize_array.shape[0]
        number_of_assets = _optimize_array.shape[1]
        if length_of_time == 1:
            best_idx = np.argmax(_optimize_array)
            weight = np.zeros(number_of_assets)
            weight[best_idx] = 1
            return weight

        # initialize weights
        weights = cp.Variable(self.number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        portfolio_return = _activation_fn * cp.log(_optimize_array * weights)

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [
                cp.sum(weights) == 1,
                weights >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
                objective=allocation_objective,
                constraints=allocation_constraints
        )

        problem.solve(warm_start=True, solver=_solver)
        return weights.value

    def optimize(self,
                 _optimize_array,
                 _solver=cp.SCS):
        """
        Calculates weights that maximize returns over a given _optimize_array

        :param _optimize_array: (np.array) relative returns of the assets for a given time period
        :param _solver: (cp.SOLVER) set the solver to be a particular cvxpy solver
        :return weights.value: (np.array) weights that maximize the returns for the given optimize_array
        """
        # calcualte length of time
        length_of_time = _optimize_array.shape[0]
        # calculate number of assets
        number_of_assets = _optimize_array.shape[1]
        # edge case to speed up calculation
        if length_of_time == 1:
            # in case that the optimize array is only one row, weights will be 1 for the highest relative return asset
            best_idx = np.argmax(_optimize_array)
            # initialize np.array of zeros
            weight = np.zeros(number_of_assets)
            # add 1 to the best performing stock
            weight[best_idx] = 1
            return weight

        # initialize weights for optimization problem
        weights = cp.Variable(self.number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        portfolio_return = cp.sum(cp.log(_optimize_array * weights))

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [cp.sum(weights) == 1, cp.min(weights) >= 0]
        # Define and solve the problem
        problem = cp.Problem(objective=allocation_objective, constraints=allocation_constraints)
        # if there is a specified solver use it
        problem.solve(warm_start=True, solver=_solver)
        return weights.value

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
