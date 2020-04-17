# pylint: disable=missing-module-docstring
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

    # adjust for previous returns
    # even if we don't rebalance, weights change because of the underlying price changes
    def update_weight(self, _weights, _relative_return, _time):
        """
        Changes step-by-step update method to adjsut for underlying asset price changes

        :param _relative_return: (np.array) previous time period's relative return
        :param _weights: (np.array) new weights that are adjusted for price change
        :param _time: (int) time for current weights
        :return
        """
        # calculate adjusted weights and then normalize
        new_weight = self.normalize(_weights * _relative_return[_time - 1])
        return new_weight
