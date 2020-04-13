# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
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
        """

        :param _weights:
        :param _relative_return:
        :return:
        """
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
        for time in range(1, self.final_number_of_time):
            # update weights
            self.weights = self.update_weight(random_portfolio, cumulative_returns, time)
            self.all_weights[time] = self.weights

    def first_weight(self, _weights):
        """

        :param _weights:
        :return:
        """
        return np.mean(_weights, axis=1)

    def update_weight(self, _random_portfolio, _cumulative_returns, _time):
        """

        :param _random_portfolio:
        :param _cumulative_returns:
        :param _time:
        :return:
        """
        return np.dot(_cumulative_returns[_time], _random_portfolio.T)


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    universal_portfolio = UP(10000)
    universal_portfolio.allocate(stock_price)
    print(universal_portfolio.all_weights)
    print(universal_portfolio.portfolio_return)
    universal_portfolio.portfolio_return.plot()


if __name__ == "__main__":
    main()
