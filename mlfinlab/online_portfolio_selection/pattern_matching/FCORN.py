# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pattern_matching.SCORN import SCORN


class FCORN(SCORN):
    """
    This class implements the Functional Correlation Driven Nonparametric Learning strategy.
    """

    def __init__(self, window, rho, lamb):
        self.lamb = lamb
        super().__init__(window=window, rho=rho)

    def update_weight(self, _weights, _relative_return, _time):
        """
        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        similar_set = []
        opposite_set = []
        new_weights = self.uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            activation_fn = np.zeros(self.final_number_of_time)
            for i in range(self.window + 1, _time - 1):
                c = self.corr_coef[i - 1][_time - 1]
                if c >= 0:
                    activation_fn[i] = self.sigmoid(-self.lamb * (c - self.rho))
                else:
                    activation_fn[i] = self.sigmoid(-self.lamb * (c + self.rho))
            new_weights = self.optimize(_relative_return, activation_fn)
        return new_weights

    def sigmoid(val):
        """
        Generates the resulting sigmoid function

        :param val: (float) input for the sigmoid function
        :return sig: (float) sigmoid(x)
        """
        res = 1 / (1 + np.exp(-val))
        return res

def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    fcorn = FCORN(window=3, rho=0.8, lamb=500)
    fcorn.allocate(stock_price, resample_by='m')
    print(fcorn.all_weights)
    print(fcorn.portfolio_return)
    fcorn.portfolio_return.plot()


if __name__ == "__main__":
    main()
