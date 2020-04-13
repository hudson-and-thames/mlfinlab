# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.OLPS import OLPS

# class to aggregate algorithms
# for example 1000 strategies involving CORN or 1000 different weights for BCRP
class AA(object):
    # what strategy to implement
    def __init__(self):
        self.strategy = None
        self.number_of_experts = None


    def allocate(self, strategy, number_of_experts):
        pass

    def generate_experts(self):
        pass

def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    agg_algorithm = AA()
    agg_algorithm.allocate(stock_price, resample_by='m')
    print(agg_algorithm.all_weights)
    print(agg_algorithm.portfolio_return)
    agg_algorithm.portfolio_return.plot()


if __name__ == "__main__":
    main()
