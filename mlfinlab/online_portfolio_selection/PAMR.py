from mlfinlab.online_portfolio_selection.olps_utils import *
import cvxpy as cp
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class PAMR(OLPS):
    """
    This class implements the Passive Aggresive Mean Reversion strategy.
    """

    def __init__(self, sensitivity=0.5, aggressiveness=1, optimization_method=0):
        """
        Constructor.
        """
        # check that sensitivity is within [0,1]
        self.sensitivity = sensitivity
        self.aggressiveness = aggressiveness
        self.optimization_method = optimization_method
        super().__init__()


    def update_weight(self, _weights, _relative_return, _time):
        # calculation prep
        _past_relative_return = _relative_return[_time - 1]
        loss = max(0, np.dot(_weights, _past_relative_return))
        adjusted_market_change = _past_relative_return - self.uniform_weight(self.number_of_assets) * np.mean(
            _past_relative_return)
        diff_norm = np.linalg.norm(adjusted_market_change)

        # different optimization methods
        if self.optimization_method == 0:
            tau = loss / (diff_norm ** 2)
        elif self.optimization_method == 1:
            tau = min(self.aggressiveness, loss / (diff_norm ** 2))
        elif self.optimization_method == 2:
            tau = loss / (diff_norm ** 2 + 1 / (2 * self.aggressiveness))

        new_weights = _weights - tau * adjusted_market_change
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
    pamr = PAMR()
    pamr.allocate(stock_price, resample_by='M')
    print(pamr.all_weights)
    print(pamr.portfolio_return)
    pamr.portfolio_return.plot()


if __name__ == "__main__":
    main()
