from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class EG(OLPS):
    """

    """

    def __init__(self, eta=0.05, update_rule='EG'):
        """
        Constructor.
        """
        super().__init__()
        self.eta = eta
        self.update_rule = update_rule

    def update_weight(self, _weights, _relative_return, _time):
        past_relative_return = _relative_return[_time - 1]
        dot_product = np.dot(_weights, past_relative_return)

        if self.update_rule == 'EG':
            new_weight = _weights * np.exp(self.eta * past_relative_return / dot_product)
        elif self.update_rule == 'GP':
            new_weight = _weights + self.eta * (past_relative_return - np.sum(past_relative_return) / self.number_of_assets) / dot_product
        elif self.update_rule == 'EM':
            new_weight = _weights * (1 + self.eta * (past_relative_return/dot_product - 1))

        return self.normalize(new_weight)


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    print("This is for EG")
    eg = EG()
    eg.allocate(stock_price)
    print(eg.all_weights)
    print(eg.portfolio_return)
    eg.portfolio_return.plot()

    print("This is for GP")
    gp = EG(update_rule='GP')
    gp.allocate(stock_price)
    print(gp.all_weights)
    print(gp.portfolio_return)
    gp.portfolio_return.plot()

    print("This is for EG")
    em = EG(update_rule='EM')
    em.allocate(stock_price)
    print(em.all_weights)
    print(em.portfolio_return)
    em.portfolio_return.plot()


if __name__ == "__main__":
    main()
