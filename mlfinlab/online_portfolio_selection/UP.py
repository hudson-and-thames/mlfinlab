from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class UP(OLPS):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def __init__(self, _number_of_iterations):
        """
        Constructor.
        """
        super(UP, self).__init__()
        self.number_of_iterations = _number_of_iterations

    def run(self, _weights, _relative_return):
        random_portfolio = self.generate_simplex(self.number_of_iterations, self.number_of_assets)

        # calculate the returns for all weights
        all_returns = np.dot(self.final_relative_return, random_portfolio)

        # calculate cumulative returns
        cumulative_returns = np.array(all_returns).cumprod(axis=0)

        # cumulative returns divided by number of random portfolios
        cumulative_returns = cumulative_returns / self.number_of_iterations

        # set initial weights
        self.weights = self.first_weight(random_portfolio)
        self.all_weights[0] = self.weights

        # Run the Algorithm for the rest of data
        for t in range(1, self.final_number_of_time):
            # update weights
            self.weights = self.update_weight(random_portfolio, cumulative_returns, t)
            self.all_weights[t] = self.weights

    def first_weight(self, _weights):
        return np.mean(_weights, axis=1)

    def update_weight(self, _random_portfolio, _cumulative_returns, _time):
        return np.dot(_cumulative_returns[_time], _random_portfolio.T)

    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    def generate_simplex(self, _number_of_portfolio, _number_of_assets):
        simplex = np.sort(np.random.random((_number_of_portfolio, _number_of_assets - 1)))
        simplex = np.diff(np.hstack([np.zeros((_number_of_portfolio,1)), simplex, np.ones((_number_of_portfolio,1))]))
        return simplex.T

def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    up = UP(10000)
    up.allocate(stock_price)
    print(up.all_weights)
    print(up.portfolio_return)
    up.portfolio_return.plot()


if __name__ == "__main__":
    main()
