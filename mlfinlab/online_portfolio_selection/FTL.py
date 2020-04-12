from mlfinlab.online_portfolio_selection import BCRP
from mlfinlab.online_portfolio_selection.olps_utils import *
import cvxpy as cp


class FTL(BCRP):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

    def update_weight(self, _weights, _relative_return, _time):
        return self.optimize(_relative_return[:_time])


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    ftl = FTL()
    ftl.allocate(stock_price)
    print(ftl.all_weights)
    print(ftl.portfolio_return)
    ftl.portfolio_return.plot()


if __name__ == "__main__":
    main()
