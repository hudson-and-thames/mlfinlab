# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning import CorrelationDrivenNonparametricLearning


class SCORN(CorrelationDrivenNonparametricLearning):
    """
    This class implements the Symmetric Correlation Driven Nonparametric Learning strategy.
    """

    # check -1 <= rho <= 1
    # check window >= 1

    def _update_weight(self, _weights, _relative_return, _time):
        """
        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        similar_set = []
        opposite_set = []
        new_weights = self._uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            activation_fn = np.zeros(self.final_number_of_time)
            for i in range(self.window + 1, _time - 1):
                if self.corr_coef[i - 1][_time - 1] > self.rho:
                    similar_set.append(i)
                elif self.corr_coef[i - 1][_time - 1] < -self.rho:
                    opposite_set.append(i)
            if similar_set:
                # put 1 for the values in the set
                activation_fn[similar_set] = 1
            if opposite_set:
                activation_fn[opposite_set] = -1
            new_weights = self._optimize(_relative_return, activation_fn)
        return new_weights

    def _optimize(self, _optimize_array, _activation_fn):
        # initial guess
        weights = self._uniform_weight(self.number_of_assets)

        def objective(_weights):
            return -np.dot(_activation_fn, np.log(np.dot(_optimize_array, _weights)))

        # weight bounds
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))
        # sum of weights = 1
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(objective, weights, method='SLSQP', bounds=bounds, constraints=const)
        return problem.x


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True,
                              index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    scorn = SCORN(window=5, rho=0.2)
    scorn.allocate(stock_price)
    print(scorn.all_weights)
    print(scorn.portfolio_return)
    scorn.portfolio_return.plot()


if __name__ == "__main__":
    main()
