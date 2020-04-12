from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class BAH(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    The Buy and Hold strategy one invests wealth among the market with an initial portfolio of weights and holds
    the portfolio till the end. The manager only buys the assets at the beginning of the first period and does
    not rebalance in subsequent periods.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

    # adjust for previous returns
    # even if we don't rebalance, weights change because of the underlying price changes
    def update_weight(self, _weights, _relative_return, _time):
        new_weight = _weights * _relative_return[_time - 1]
        return self.normalize(new_weight)

def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    bah = BAH()
    bah.allocate(stock_price)
    print(bah.all_weights)
    print(bah.portfolio_return)
    bah.portfolio_return.plot()


if __name__ == "__main__":
    main()