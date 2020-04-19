# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CORN(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy.
    """
    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, window, rho):
        """
        Constructor.
        """
        self.window = window
        self.rho = rho
        self.corr_coef = None
        super().__init__()

    # calculate corr_coef ahead of updating to speed up calculations
    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        """
        :param _asset_prices:
        :param _weights:
        :param _portfolio_start:
        :param _resample_by:
        :return:
        """
        super(CORN, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)
        self.corr_coef = self.calculate_rolling_correlation_coefficient(self.final_relative_return)

    def update_weight(self, _weights, _relative_return, _time):
        """
        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        similar_set = []
        new_weights = self.uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            activation_fn = np.zeros(self.final_number_of_time)
            for i in range(self.window + 1, _time - 1):
                if self.corr_coef[i - 1][_time - 1] > self.rho:
                    similar_set.append(i)
            if similar_set:
                # put 1 for the values in the set
                activation_fn[similar_set] = 1
                new_weights = self.optimize(_relative_return, activation_fn, cp.SCS)
        return new_weights

    # optimize the weight that maximizes the returns
    def optimize(self, _optimize_array, _activation_fn, _solver=None):
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


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn = CORN(window=1, rho=0.1)
    corn.allocate(stock_price, resample_by='w')
    print(corn.all_weights)
    print(corn.portfolio_return)
    corn.portfolio_return.plot()


if __name__ == "__main__":
    main()
