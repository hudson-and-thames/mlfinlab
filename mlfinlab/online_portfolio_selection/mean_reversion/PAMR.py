# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class PAMR(OLPS):
    """
    This class implements the Passive Aggresive Mean Reversion strategy.
    """

    def __init__(self, sensitivity=0.5, aggressiveness=1, optimization_method=0):
        """
        Constructor.
        """
        # check that sensitivity is within [0,1]
        self.sensitivity = sensitivity
        self.aggressiveness = aggressiveness
        self.optimization_method = optimization_method
        super().__init__()


    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        # calculation prep
        _past_relative_return = _relative_return[_time - 1]
        loss = max(0, np.dot(_weights, _past_relative_return))
        adjusted_market_change = _past_relative_return - self.uniform_weight(self.number_of_assets) * np.mean(
            _past_relative_return)
        diff_norm = np.linalg.norm(adjusted_market_change)

        # different optimization methods
        if self.optimization_method == 0:
            tau = loss / (diff_norm ** 2)
        elif self.optimization_method == 1:
            tau = min(self.aggressiveness, loss / (diff_norm ** 2))
        elif self.optimization_method == 2:
            tau = loss / (diff_norm ** 2 + 1 / (2 * self.aggressiveness))

        new_weights = _weights - tau * adjusted_market_change
        # if not in simplex domain
        return self.simplex_projection(new_weights)


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    pamr = PAMR()
    pamr.allocate(stock_price, resample_by='M')
    print(pamr.all_weights)
    print(pamr.portfolio_return)
    pamr.portfolio_return.plot()


if __name__ == "__main__":
    main()
