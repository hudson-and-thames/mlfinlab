# pylint: disable=missing-module-docstring
import pandas as pd
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class ConfidenceWeightedMeanReversion(OLPS):
    """
    This class implements the Confidence Weighted Mean Reversion strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    Online Moving Average Reversion reverts to the SMA or EMA of the underlying assets based on the given threshold.
    """

    def __init__(self,
                 confidence=0.5,
                 epsilon=0.5):
        """
        Constructor.
        """
        self.confidence = confidence
        self.epsilon = epsilon
        super().__init__()

    # will update later

    # def update_weight(self, _weights, _relative_return, _time):
    #     # calculation prep
    #     _past_relative_return = _relative_return[_time - 1]
    #     loss = max(0, np.dot(_weights, _past_relative_return))
    #     adjusted_market_change = _past_relative_return - self.uniform_weight(self.number_of_assets) * np.mean(
    #         _past_relative_return)
    #     diff_norm = np.linalg.norm(adjusted_market_change)
    #
    #     # different optimization methods
    #     if self.optimization_method == 0:
    #         tau = loss / (diff_norm ** 2)
    #     elif self.optimization_method == 1:
    #         tau = min(self.aggressiveness, loss / (diff_norm ** 2))
    #     elif self.optimization_method == 2:
    #         tau = loss / (diff_norm ** 2 + 1 / (2 * self.aggressiveness))
    #
    #     new_weights = _weights - tau * adjusted_market_change
    #     # if not in simplex domain
    #     if ((new_weights > 1) | (new_weights < 0)).any():
    #         return self.optimize(new_weights)
    #     else:
    #         return new_weights

def main():
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    cwmr = ConfidenceWeightedMeanReversion()
    cwmr.allocate(stock_price, resample_by='M')
    print(cwmr.all_weights)
    print(cwmr.portfolio_return)
    cwmr.portfolio_return.plot()


if __name__ == "__main__":
    main()
