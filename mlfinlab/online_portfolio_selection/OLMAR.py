from mlfinlab.online_portfolio_selection.olps_utils import *
import cvxpy as cp
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class OLMAR(OLPS):
    """
    This class implements the Online Moving Average Reversion Strategy.
    """

    def __init__(self, epsilon=2, window=2, alpha=.9, reversion_method=1):
        """
        Constructor.
        """
        # check that epsilon is > 1
        # check that window is >= 1
        # check that alpha is (0,1)
        # check that reversion_method is either 1 or 2
        # if optimization_method == 2 then reversion_method doesn't matter
        self.epsilon = epsilon
        self.window = window
        self.alpha = alpha
        self.reversion_method = reversion_method
        self.moving_average_reversion = None
        super().__init__()

    # intialize moving average reversion
    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        self.moving_average_reversion = self.calculate_rolling_moving_average(_asset_prices, self.window,
                                                                              self.reversion_method, self.alpha)
        super(OLMAR, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)

    # change based on method
    def run(self, _weights, _relative_return):
        # set initial weights
        self.weights = self.first_weight(_weights)
        self.all_weights[0] = self.weights

        # Run the Algorithm for the rest of data
        for t in range(1, self.final_number_of_time):
            # update weights
            self.weights = self.update_weight(self.weights, _relative_return, t)
            self.all_weights[t] = self.weights

    def update_weight(self, _weights, _relative_return, _time):
        if self.reversion_method == 1 and _time < self.window:
            return self.weights
        # calculate price relative
        predicted_change = self.moving_average_reversion[_time]
        # different OLMAR methods
        mean_relative = np.mean(predicted_change)
        mean_change = np.ones(self.number_of_assets) * mean_relative
        lambd = max(0, (self.epsilon - np.dot(_weights, predicted_change)) / (
                    np.linalg.norm(predicted_change - mean_change) ** 2))

        new_weights = _weights + lambd * (predicted_change - mean_change)
        if np.isnan(new_weights).any():
            raise ValueError()
        # if not in simplex domain
        if ((new_weights > 1) | (new_weights < 0)).any():
            return self.optimize(new_weights)
        else:
            return new_weights

    # optimize the weight that minimizes the l2 norm
    def optimize(self, _optimize_weight):
        # initialize weights
        weights = cp.Variable(self.number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        l2_norm = cp.norm(weights - _optimize_weight)

        # Optimization objective and constraints
        allocation_objective = cp.Minimize(l2_norm)
        allocation_constraints = [
                cp.sum(weights) == 1,
                weights <= 1,
                weights >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
                objective=allocation_objective,
                constraints=allocation_constraints
        )
        problem.solve(solver=cp.SCS)
        return weights.value


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)

    olmar1 = OLMAR(epsilon=2, window=7, reversion_method=1)
    olmar1.allocate(stock_price, resample_by='M')
    print(olmar1.all_weights)
    print(olmar1.portfolio_return)
    olmar1.portfolio_return.plot()

    olmar2 = OLMAR(epsilon=2, alpha=.9, reversion_method=2)
    olmar2.allocate(stock_price, resample_by='M')
    print(olmar2.all_weights)
    print(olmar2.portfolio_return)
    olmar2.portfolio_return.plot()


if __name__ == "__main__":
    main()
