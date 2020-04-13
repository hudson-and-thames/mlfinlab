# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CORN(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy.
    """
    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, window=20, rho=0.6):
        """
        Constructor.
        """
        self.window = window
        self.rho = rho
        self.corr_coef = None
        super().__init__()

    # calculate corr_coef ahead of updating to speed up calculations
    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        """
        :param _asset_prices:
        :param _weights:
        :param _portfolio_start:
        :param _resample_by:
        :return:
        """
        super(CORN, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)
        self.corr_coef = self.calculate_rolling_correlation_coefficient(self.final_relative_return)

    def update_weight(self, _weights, _relative_return, _time):
        """
        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        similar_set = []
        new_weights = self.uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            for i in range(self.window + 1, _time - 1):
                if self.corr_coef[i - 1][_time - 1] > self.rho:
                    similar_set.append(i)
            if similar_set:
                similar_sequences = _relative_return[similar_set]
                new_weights = self.optimize(similar_sequences, cp.SCS)
        return new_weights

    def generate_random_window_rho(self, _length_of_window, _number_of_rho):
        """
        :param _length_of_window:
        :param _number_of_rho:
        :return:
        """


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn = CORN()
    corn.allocate(stock_price, resample_by='m')
    print(corn.all_weights)
    print(corn.portfolio_return)
    corn.portfolio_return.plot()


if __name__ == "__main__":
    main()
