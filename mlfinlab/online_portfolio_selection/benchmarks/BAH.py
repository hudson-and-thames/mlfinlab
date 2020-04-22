# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class BAH(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    The Buy and Hold strategy invests capital with an initial portfolio of weights and holds the portfolio till the
    end. The manager only buys the assets at the beginning of the first period and does not rebalance in subsequent
    periods.
    """

    def update_weight(self,
                      _time):
        """
        Changes step-by-step update method to adjsut for underlying asset price changes

        :param _time: (int) current time period
        :return (None) adjusts weights for previous time's price changes
        """
        # return same weights because price has not changed
        if _time == 0:
            return self.weights
        # adjust it for price differences
        return self.weights * self.relative_return[_time - 1]
