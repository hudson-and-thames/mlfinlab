# pylint: disable=missing-module-docstring
import pandas as pd
from mlfinlab.online_portfolio_selection import BCRP


class FTL(BCRP):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        return self.optimize(_relative_return[:_time])


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    ftl = FTL()
    ftl.allocate(stock_price)
    print(ftl.all_weights)
    print(ftl.portfolio_return)
    ftl.portfolio_return.plot()


if __name__ == "__main__":
    main()
