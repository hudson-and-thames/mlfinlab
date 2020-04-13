# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection import OLPS
import pandas as pd


class BCRP(OLPS):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def first_weight(self, _weights):
        """
        :param _weights: optimized first weight
        :return:
        """
        return self.optimize(self.final_relative_return)

def main():
    """
    test run
    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    bcrp = BCRP()
    bcrp.allocate(stock_price)
    print(bcrp.all_weights)
    print(bcrp.portfolio_return)
    bcrp.portfolio_return.plot()


if __name__ == "__main__":
    main()
