# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class EG(OLPS):
    """
    Exponential Gradient
    """

    def __init__(self, eta=0.05, update_rule='EG'):
        """
        Constructor.
        """
        super().__init__()
        self.eta = eta
        self.update_rule = update_rule

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        past_relative_return = _relative_return[_time - 1]
        dot_product = np.dot(_weights, past_relative_return)

        if self.update_rule == 'EG':
            new_weight = _weights * np.exp(self.eta * past_relative_return / dot_product)
        elif self.update_rule == 'GP':
            new_weight = _weights + self.eta * (past_relative_return - np.sum(past_relative_return) / self.number_of_assets) / dot_product
        elif self.update_rule == 'EM':
            new_weight = _weights * (1 + self.eta * (past_relative_return/dot_product - 1))

        return self.normalize(new_weight)


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    print("This is for EG")
    exponential_gradient = EG()
    exponential_gradient.allocate(stock_price)
    print(exponential_gradient.all_weights)
    print(exponential_gradient.portfolio_return)
    exponential_gradient.portfolio_return.plot()

    print("This is for GP")
    gradient_projection = EG(update_rule='GP')
    gradient_projection.allocate(stock_price)
    print(gradient_projection.all_weights)
    print(gradient_projection.portfolio_return)
    gradient_projection.portfolio_return.plot()

    print("This is for EG")
    expectation_maximization = EG(update_rule='EM')
    expectation_maximization.allocate(stock_price)
    print(expectation_maximization.all_weights)
    print(expectation_maximization.portfolio_return)
    expectation_maximization.portfolio_return.plot()


if __name__ == "__main__":
    main()
