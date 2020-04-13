# pylint: disable=missing-module-docstring
import pandas as pd
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CRP(OLPS):
    """
    Basically OLPS
    """

def main():
    """
    test run
    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    crp = CRP()
    crp.allocate(stock_price)
    print(crp.all_weights)
    print(crp.portfolio_return)
    crp.portfolio_return.plot()


if __name__ == "__main__":
    main()
