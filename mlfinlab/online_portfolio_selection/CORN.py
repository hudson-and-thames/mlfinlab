from mlfinlab.online_portfolio_selection.olps_utils import *
import cvxpy as cp
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CORN(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.window = None
        self.rho = None
        self.corr_coef = None
        super().__init__()
    
    def allocate(self,
                 asset_prices,
                 weights=None,
                 window=20,
                 rho=.6,
                 portfolio_start=0,
                 resample_by=None):
        self.window = window
        self.rho = rho
        super(CORN, self).allocate(asset_prices, weights, portfolio_start, resample_by)

    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        super(CORN, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)
        self.corr_coef = self.calculate_rolling_correlation_coefficient(self.final_relative_return)

    def update_weight(self, _weights, _relative_return, _time):
        similar_set = []
        new_weights = self.uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            for i in range(self.window + 1, _time - 1):
                if self.corr_coef[i - 1][_time - 1] > self.rho:
                    similar_set.append(i)
            if similar_set:
                similar_sequences = _relative_return[similar_set]
                new_weights = self.optimize(similar_sequences)
        return new_weights

    # optimize the weight that maximizes the returns
    def optimize(self, _optimize_array):
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
        portfolio_return = cp.sum(cp.log(_optimize_array * weights))

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
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
    corn = CORN()
    corn.allocate(stock_price, resample_by='w')
    print(corn.all_weights)
    print(corn.portfolio_return)
    corn.portfolio_return.plot()


if __name__ == "__main__":
    main()
